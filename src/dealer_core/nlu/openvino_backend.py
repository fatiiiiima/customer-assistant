"""OpenVINO GenAI NLU backend (voice path). Device GPU = the Arc iGPU, CPU = no-iGPU fallback.

Notes from Phase 0b measurements:
  * OpenVINO reuses its KV cache across generate() calls (a repeated prefix is cheap on CPU, a miss is not).
  * Never pin INFERENCE_NUM_THREADS on this hybrid-core CPU: it made things ~1.5x slower.
  * The NPU cannot compile the pre-converted int4 LLMs, so it is not offered here.
"""
from __future__ import annotations

import json
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Iterator

from ..config import BackendCfg, LlmCfg
from ..dialogue.states import Intent, State
from . import prompts
from .backend import BackendError, BackendHealth, BackendTimeout, IntentResult, NluBackend, resolve_prompt_mode

_SENTENCE_END = re.compile(r"[.!?]\s*$")


class OpenVinoBackend(NluBackend):
    def __init__(self, name: str, cfg: BackendCfg, llm: LlmCfg, model_dir: Path) -> None:
        self.name, self.cfg, self.llm = name, cfg, llm
        self.mode = resolve_prompt_mode(cfg)
        self.path = model_dir / "ov" / cfg.model
        self.cache_dir = model_dir / "ov_cache"
        self._pipe = None
        self._tok = None
        self._configs: dict[tuple[str, str], object] = {}
        # One worker: the pipeline is not re-entrant, and it gives us a timeout without killing the process.
        # With batching, many requests run together (the batching thread steps them all), so the pool only waits on them.
        self._pool = ThreadPoolExecutor(max_workers=cfg.max_parallel if cfg.batching else 1, thread_name_prefix=f"ov-{name}")

    # -- lifecycle ---------------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        import openvino_genai as ov  # lazy

        if not self.path.exists():
            raise BackendError(f"OpenVINO model folder missing: {self.path}")
        try:
            if self.cfg.batching:
                from .ov_batching import BatchedPipeline

                self._pipe = BatchedPipeline(self.path, self.cfg.device or "CPU", self.cache_dir, kv_cache_gb=self.cfg.kv_cache_gb)
            else:
                self._pipe = ov.LLMPipeline(str(self.path), self.cfg.device or "CPU", CACHE_DIR=str(self.cache_dir))
        except Exception as e:  # device missing, compile failure
            raise BackendError(f"cannot load {self.cfg.model} on {self.cfg.device}: {e}") from e
        self._tok = self._pipe.get_tokenizer()
        self._ov = ov

    def health(self) -> BackendHealth:
        try:
            self._ensure_loaded()
        except BackendError as e:
            return BackendHealth(False, str(e))
        return BackendHealth(True, f"openvino {self.cfg.model}@{self.cfg.device} ({self.mode} prompt)")

    # -- internals ---------------------------------------------------------------------------
    def _prompt(self, system: str, user: str) -> str:
        hybrid = not self.llm.think and self.path.name.startswith("Qwen3-") and "Instruct-2507" not in self.path.name
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user + (" /no_think" if hybrid else "")}]
        if self.path.name.lower().startswith("mistral"):           # Mistral's chat template has no system role: the instructions open the user turn
            messages = [{"role": "user", "content": system + "\n\n" + messages[1]["content"]}]
        prompt = self._tok.apply_chat_template(messages, add_generation_prompt=True, extra_context={"enable_thinking": self.llm.think})
        if hybrid and not prompt.rstrip().endswith("</think>"):
            prompt += "<think>\n\n</think>\n\n"           # hybrid Qwen3 models (8B): the converted template ignores enable_thinking, so close the thinking block ourselves
        return prompt

    def _gen_config(self, kind: str, state: State):
        key = (kind, state.value)
        if key not in self._configs:
            cfg = self._ov.GenerationConfig()
            cfg.do_sample = False                      # greedy == temperature 0
            cfg.max_new_tokens = self.cfg.max_intent_tokens if kind == "intent" else self.cfg.max_json_tokens
            schema = prompts.intent_only_schema(state) if kind == "intent" else prompts.full_schema(state)
            cfg.structured_output_config = self._ov.StructuredOutputConfig(json_schema=json.dumps(schema))
            self._configs[key] = cfg
        return self._configs[key]

    def _run(self, kind: str, state: State, car: str | None, text: str, last_agent: str = "") -> tuple[dict, str, float]:
        self._ensure_loaded()
        prompt = self._prompt(prompts.system_prompt(state, self.mode), prompts.user_message(state, car, text, last_agent))
        cfg = self._gen_config(kind, state)
        t0 = time.perf_counter()
        fut = self._pool.submit(lambda: str(self._pipe.generate([prompt], cfg).texts[0]))
        try:
            raw = fut.result(timeout=self.cfg.timeout_s)
        except FutureTimeout as e:
            raise BackendTimeout(f"{self.name} exceeded {self.cfg.timeout_s}s") from e
        except Exception as e:
            raise BackendError(f"{type(e).__name__}: {e}") from e
        try:
            return json.loads(raw), raw, (time.perf_counter() - t0) * 1000
        except json.JSONDecodeError as e:
            raise BackendError(f"malformed JSON from {self.name}: {raw!r}") from e

    def warm_up(self) -> None:
        """First use of each state compiles its grammar (several seconds). Do it for every state up front, with no timeout."""
        from ..dialogue.states import nlu_states

        self._ensure_loaded()
        for state in nlu_states():
            prompt = self._prompt(prompts.system_prompt(state, self.mode), prompts.user_message(state, None, "hello there"))
            self._pool.submit(lambda p=prompt, c=self._gen_config("intent", state): self._pipe.generate([p], c)).result()

    # -- interface ---------------------------------------------------------------------------
    def classify(self, state: State, car: str | None, text: str, last_agent: str = "") -> IntentResult:
        obj, raw, ms = self._run("intent", state, car, text, last_agent)
        return IntentResult(Intent(obj["intent"]), ms, self.name, raw)

    def lead_in(self, system: str, user: str, max_tokens: int = 36, temperature: float | None = None, deadline_s: float | None = None) -> str:
        """Free text. With `deadline_s` the generation is STOPPED when the time is up and BackendTimeout is raised (no partial text)."""
        from ..dialogue.leadin import LEAD_TIMEOUT_S

        self._ensure_loaded()
        ov = self._ov
        cfg = ov.GenerationConfig()
        cfg.do_sample, cfg.max_new_tokens = False, max_tokens
        if temperature:                                    # free-text replies are sampled so the wording varies; intents stay greedy
            cfg.do_sample, cfg.temperature, cfg.top_p, cfg.top_k = True, temperature, 0.9, 40
        prompt = self._prompt(system, user)
        limit = deadline_s or LEAD_TIMEOUT_S * (3.5 if max_tokens > 80 else 2 if max_tokens > 36 else 1)
        started = time.perf_counter()
        timed_out = threading.Event()

        def streamer(_piece: str):
            if time.perf_counter() - started > limit:
                timed_out.set()
                return ov.StreamingStatus.STOP
            return ov.StreamingStatus.RUNNING

        fut = self._pool.submit(lambda: str(self._pipe.generate(prompt, cfg, streamer)))
        try:
            text = fut.result(timeout=limit + 5.0)         # the streamer stops generation at `limit`; this is only a safety net
        except FutureTimeout as e:
            raise BackendTimeout(f"{self.name} lead-in exceeded {limit}s") from e
        if timed_out.is_set():
            raise BackendTimeout(f"{self.name} lead-in stopped at its {limit:.1f}s deadline")
        return re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.S)       # hybrid models: drop the (empty) thinking block

    def lead_in_stream(self, system: str, user: str, max_tokens: int = 36, temperature: float | None = None,
                       deadline_s: float | None = None, prefill: str = "") -> Iterator[str]:
        """Pieces of free text as the model produces them. Generation is STOPPED when the consumer stops iterating or the deadline passes
        (the iterator then simply ends: what was already yielded stands)."""
        from ..dialogue.leadin import LEAD_TIMEOUT_S

        self._ensure_loaded()
        ov = self._ov
        cfg = ov.GenerationConfig()
        cfg.do_sample, cfg.max_new_tokens = False, max_tokens
        if temperature:
            cfg.do_sample, cfg.temperature, cfg.top_p, cfg.top_k = True, temperature, 0.9, 40
        prompt = self._prompt(system, user) + prefill            # the assistant's turn already begun by code: the model only continues it
        limit = deadline_s or LEAD_TIMEOUT_S * 3
        started = time.perf_counter()
        pieces: queue.Queue = queue.Queue()
        stop = threading.Event()

        def streamer(piece: str):
            pieces.put(piece)
            return ov.StreamingStatus.STOP if (stop.is_set() or time.perf_counter() - started > limit) else ov.StreamingStatus.RUNNING

        def work():
            try:
                self._pipe.generate(prompt, cfg, streamer)
            except Exception as e:                               # surfaced to the consumer below
                pieces.put(e)
            finally:
                pieces.put(None)

        self._pool.submit(work)
        head, think_done = "", False
        try:
            while True:
                try:
                    item = pieces.get(timeout=limit + 5.0)
                except queue.Empty as e:
                    raise BackendTimeout(f"{self.name} produced nothing within {limit + 5:.0f}s") from e
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise BackendError(f"{type(item).__name__}: {item}") from item
                if not think_done:                                           # hybrid models start with an (empty) <think></think> block: not speech
                    head += item
                    if "</think>" in head:
                        think_done, item = True, head.split("</think>", 1)[1].lstrip()
                    elif len(head) > 12 and not head.lstrip().startswith("<think"):
                        think_done, item = True, head
                    else:
                        continue
                    if not item:
                        continue
                yield item
        finally:
            stop.set()

    def complete_json(self, state: State, car: str | None, text: str) -> dict:
        return self._run("json", state, car, text)[0]

    def answer(self, question: str, context: str, max_sentences: int = 2) -> Iterator[str]:
        self._ensure_loaded()
        ov = self._ov
        prompt = self._prompt(prompts.OPEN_ANSWER_SYSTEM.format(n=max_sentences), f"CONTEXT: {context}\nCALLER: {question}")
        cfg = ov.GenerationConfig()
        cfg.do_sample, cfg.max_new_tokens = False, self.cfg.max_answer_tokens
        pieces: queue.Queue = queue.Queue()
        stop = threading.Event()

        def streamer(piece: str):
            pieces.put(piece)
            return ov.StreamingStatus.STOP if stop.is_set() else ov.StreamingStatus.RUNNING

        def work():
            try:
                self._pipe.generate(prompt, cfg, streamer)
            finally:
                pieces.put(None)

        self._pool.submit(work)
        buf, sentences = "", 0
        try:
            while (piece := pieces.get(timeout=self.cfg.timeout_s * 4)) is not None:
                buf += piece
                if _SENTENCE_END.search(buf) and len(buf.split()) >= 3:
                    yield buf.strip()
                    buf, sentences = "", sentences + 1
                    if sentences >= max_sentences:
                        return
            if buf.strip():
                yield buf.strip()
        except queue.Empty as e:
            raise BackendError(f"{self.name} produced no output within the answer timeout") from e
        finally:
            stop.set()   # tells the streamer to stop generating if the caller abandoned the iterator
