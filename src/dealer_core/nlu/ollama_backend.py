"""Ollama NLU backend (text core, website chat, and the voice-path fallback)."""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Iterator

from ..config import BackendCfg, LlmCfg
from ..dialogue.states import Intent, State
from . import prompts
from .backend import BackendError, BackendHealth, BackendTimeout, IntentResult, NluBackend, resolve_prompt_mode

_SENTENCE_END = re.compile(r"[.!?]\s*$")


class OllamaBackend(NluBackend):
    def __init__(self, name: str, cfg: BackendCfg, llm: LlmCfg) -> None:
        import ollama  # lazy: the package is only needed when this backend is used

        self.name, self.cfg, self.llm = name, cfg, llm
        self.mode = resolve_prompt_mode(cfg)
        self._ollama = ollama
        self.client = ollama.Client(host=cfg.host, timeout=cfg.timeout_s)

    def health(self) -> BackendHealth:
        try:
            names = {m.model for m in self.client.list().models}
        except Exception as e:  # server down, wrong host...
            return BackendHealth(False, f"ollama unreachable at {self.cfg.host}: {e}")
        if self.cfg.model not in names:
            return BackendHealth(False, f"model {self.cfg.model!r} not pulled (run: ollama pull {self.cfg.model})")
        return BackendHealth(True, f"ollama {self.cfg.model}")

    # -- internals ---------------------------------------------------------------------------
    def _chat(self, system: str, user: str, schema: dict | None, max_tokens: int, stream: bool = False, temperature: float | None = None):
        options = {"temperature": self.llm.temperature if temperature is None else temperature, "num_ctx": self.cfg.num_ctx, "num_predict": max_tokens}
        if self.cfg.num_thread:
            options["num_thread"] = self.cfg.num_thread
        kwargs = dict(model=self.cfg.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                      options=options, keep_alive=self.cfg.keep_alive, stream=stream)
        if schema is not None:
            kwargs["format"] = schema
        try:
            # think=False is Ollama's switch for Qwen3-style models. Models without a thinking mode reject it.
            return self.client.chat(think=self.llm.think, **kwargs)
        except self._ollama.ResponseError as e:
            if "think" not in str(e).lower():
                raise BackendError(str(e)) from e
            return self.client.chat(**kwargs)
        except Exception as e:  # connection error, timeout
            if "timed out" in str(e).lower() or type(e).__name__.endswith("Timeout"):
                raise BackendTimeout(f"{self.name} exceeded {self.cfg.timeout_s}s") from e
            raise BackendError(f"{type(e).__name__}: {e}") from e

    def _json(self, state: State, car: str | None, text: str, schema: dict, max_tokens: int, last_agent: str = "") -> tuple[dict, str, float]:
        t0 = time.perf_counter()
        r = self._chat(prompts.system_prompt(state, self.mode), prompts.user_message(state, car, text, last_agent), schema, max_tokens)
        raw = r.message.content or ""
        try:
            return json.loads(raw), raw, (time.perf_counter() - t0) * 1000
        except json.JSONDecodeError as e:
            raise BackendError(f"malformed JSON from {self.name}: {raw!r}") from e

    def warm_up(self) -> None:
        """Load the model into memory (a cold 4B load takes ~10 s) with a generous timeout, then keep it warm."""
        warm = self._ollama.Client(host=self.cfg.host, timeout=max(60.0, self.cfg.timeout_s * 5))
        warm.chat(model=self.cfg.model, messages=[{"role": "user", "content": "hi"}], think=False, keep_alive=self.cfg.keep_alive,
                  options={"num_predict": 1, "num_ctx": self.cfg.num_ctx})

    # -- interface ---------------------------------------------------------------------------
    def classify(self, state: State, car: str | None, text: str, last_agent: str = "") -> IntentResult:
        obj, raw, ms = self._json(state, car, text, prompts.intent_only_schema(state), self.cfg.max_intent_tokens, last_agent)
        return IntentResult(Intent(obj["intent"]), ms, self.name, raw)

    def complete_json(self, state: State, car: str | None, text: str) -> dict:
        return self._json(state, car, text, prompts.full_schema(state), self.cfg.max_json_tokens)[0]

    def lead_in(self, system: str, user: str, max_tokens: int = 36, temperature: float | None = None, deadline_s: float | None = None) -> str:
        if not deadline_s:
            return self._chat(system, user, None, max_tokens, temperature=temperature).message.content or ""
        return self._within(system, user, max_tokens, temperature, deadline_s)

    def structured(self, system: str, user: str, schema: dict, max_tokens: int = 120, deadline_s: float | None = None) -> str | None:
        """One JSON object that follows `schema` (Ollama's `format`: the answer is held to the schema by a grammar). Greedy. Past `deadline_s`: BackendTimeout, and the generation stops."""
        if not deadline_s:
            return self._chat(system, user, schema, max_tokens, temperature=0.0).message.content or ""
        return self._within(system, user, max_tokens, 0.0, deadline_s, schema=schema)

    def _within(self, system: str, user: str, max_tokens: int, temperature: float | None, deadline_s: float, schema: dict | None = None) -> str:
        """The answer, or BackendTimeout when it is not complete within `deadline_s` (28 Oct review, B2: only the 20 s client timeout applied on Ollama, so the 1.2 s decision
        deadline and the reply deadline of the vLLM path were never enforced here). The call runs in a helper thread that reads the reply as a stream and stops reading when told
        to, which closes the connection and so stops the generation on the server; Ollama has one slot here, and a late answer nobody wants would hold up the next turn."""
        parts: list[str] = []
        outcome: dict = {}
        done, stop = threading.Event(), threading.Event()

        def run() -> None:
            try:
                for chunk in self._chat(system, user, schema, max_tokens, stream=True, temperature=temperature):
                    parts.append(chunk.message.content or "")
                    if stop.is_set():
                        break
            except Exception as e:
                outcome["error"] = e
            finally:
                done.set()

        threading.Thread(target=run, daemon=True, name="ollama-bounded").start()
        if not done.wait(deadline_s):
            stop.set()
            raise BackendTimeout(f"{self.name} had no complete answer within {deadline_s:.1f}s")
        if "error" in outcome:
            raise outcome["error"]
        return "".join(parts)

    def lead_in_stream(self, system: str, user: str, max_tokens: int = 36, temperature: float | None = None, deadline_s: float | None = None, prefill: str = "") -> Iterator[str]:
        """The reply as the model writes it, so the first words can be spoken after the first clause, not after the whole reply (27-28 Oct: without this a model-written reply
        was generated whole first, 1.8-2.5 s, before anything of it was spoken). Ollama's chat cannot continue a reply already begun by code, so with a `prefill` the base
        class's way is used (it tells the model what was already said)."""
        if prefill:
            yield from super().lead_in_stream(system, user, max_tokens, temperature, deadline_s, prefill=prefill)
            return
        started = time.perf_counter()
        for chunk in self._chat(system, user, None, max_tokens, stream=True, temperature=temperature):
            piece = chunk.message.content or ""
            if piece:
                yield piece
            if deadline_s and time.perf_counter() - started > deadline_s:
                return                                          # the reply deadline (as the vLLM path does): stop here, which closes the connection and stops the model

    def answer(self, question: str, context: str, max_sentences: int = 2) -> Iterator[str]:
        system = prompts.OPEN_ANSWER_SYSTEM.format(n=max_sentences)
        stream = self._chat(system, f"CONTEXT: {context}\nCALLER: {question}", None, self.cfg.max_answer_tokens, stream=True)
        buf, sentences = "", 0
        for chunk in stream:
            buf += chunk.message.content or ""
            if _SENTENCE_END.search(buf) and len(buf.split()) >= 3:
                yield buf.strip()
                buf, sentences = "", sentences + 1
                if sentences >= max_sentences:
                    return
        if buf.strip():
            yield buf.strip()
