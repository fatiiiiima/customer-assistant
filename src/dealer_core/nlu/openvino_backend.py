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
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"ov-{name}")

    # -- lifecycle ---------------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        import openvino_genai as ov  # lazy

        if not self.path.exists():
            raise BackendError(f"OpenVINO model folder missing: {self.path}")
        try:
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
        return self._tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True, extra_context={"enable_thinking": self.llm.think})

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

    def _run(self, kind: str, state: State, car: str | None, text: str) -> tuple[dict, str, float]:
        self._ensure_loaded()
        prompt = self._prompt(prompts.system_prompt(state, self.mode), prompts.user_message(state, car, text))
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
    def classify(self, state: State, car: str | None, text: str) -> IntentResult:
        obj, raw, ms = self._run("intent", state, car, text)
        return IntentResult(Intent(obj["intent"]), ms, self.name, raw)

    def lead_in(self, system: str, user: str) -> str:
        from ..dialogue.leadin import LEAD_TIMEOUT_S

        self._ensure_loaded()
        cfg = self._ov.GenerationConfig()
        cfg.do_sample, cfg.max_new_tokens = False, 36
        prompt = self._prompt(system, user)
        fut = self._pool.submit(lambda: str(self._pipe.generate([prompt], cfg).texts[0]))
        try:
            return fut.result(timeout=LEAD_TIMEOUT_S)
        except FutureTimeout as e:
            raise BackendTimeout(f"{self.name} lead-in exceeded {LEAD_TIMEOUT_S}s") from e

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
