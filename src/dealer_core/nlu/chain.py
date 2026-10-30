"""Backend chain: use the first healthy backend, fall back on failure, and say so in the log."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Iterator

from ..config import ALLOW_CLOUD_ENV, BackendCfg, Config, LlmCfg
from ..dialogue.states import State
from .backend import BackendError, BackendTimeout, IntentResult, NluBackend

log = logging.getLogger("dealer.nlu")


def make_backend(name: str, b: BackendCfg, llm: LlmCfg, model_dir: Path) -> NluBackend:
    """Instantiate a backend from its config entry. Imports stay lazy so unused runtimes need not be installed.
    A cloud backend (the caller's words, and name and phone once given, leave the company) is refused unless VOICE_AGENT_ALLOW_CLOUD=1."""
    if b.cloud and os.environ.get(ALLOW_CLOUD_ENV) != "1":
        raise BackendError(f"backend {name!r} is a third-party cloud service: it needs {ALLOW_CLOUD_ENV}=1 (dev only)")
    if b.runtime == "ollama":
        from .ollama_backend import OllamaBackend

        return OllamaBackend(name, b, llm)
    if b.runtime == "openai":
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(name, b, llm)
    if b.runtime == "openvino":
        from .openvino_backend import OpenVinoBackend

        return OpenVinoBackend(name, b, llm, model_dir)
    raise BackendError(f"backend {name!r} has runtime {b.runtime!r}, which is a disabled stub")


_BUILT: dict[tuple, NluBackend] = {}


def build_backend(name: str, cfg: Config) -> NluBackend:
    """One instance per configured backend: the intent chain and the reply writer used to load the same model twice (2 x 4.5 GB, and two
    schedulers fighting over the iGPU)."""
    b = cfg.llm.backends[name]
    key = (name, b.model_dump_json(), str(cfg.app.model_dir))
    if key not in _BUILT:
        _BUILT[key] = make_backend(name, b, cfg.llm, Path(cfg.app.model_dir))
    return _BUILT[key]


class BackendChain:
    """Ordered backends. `start()` picks the first healthy one; calls fall through to the next on BackendError."""

    def __init__(self, backends: list[NluBackend]) -> None:
        if not backends:
            raise ValueError("at least one backend is required")
        self.backends = backends
        self.active = 0

    @classmethod
    def from_config(cls, cfg: Config, primary: str | None = None,
                    factory: Callable[[str, Config], NluBackend] = build_backend) -> "BackendChain":
        built: list[NluBackend] = []
        for name in cfg.llm.chain(primary):
            try:
                built.append(factory(name, cfg))
            except Exception as e:  # missing package, stub, bad config
                log.warning("backend %s unavailable: %s", name, e)
        return cls(built)

    def start(self, warm: bool = True) -> str:
        """Health-check in order; the first healthy backend becomes active and is warmed up (grammars compiled, weights loaded)
        so the first caller does not hit a cold start. Fallbacks are NOT loaded. Returns a one-line status for the log."""
        reasons = []
        for i, b in enumerate(self.backends):
            h = b.health()
            if h.ok:
                self.active = i
                if warm:
                    b.warm_up()
                msg = f"NLU backend active: {b.name} ({h.detail})" + (f"; skipped: {'; '.join(reasons)}" if reasons else "")
                log.info(msg)
                return msg
            reasons.append(f"{b.name}: {h.detail}")
            log.warning("backend %s unhealthy: %s", b.name, h.detail)
        raise BackendError("no healthy NLU backend: " + "; ".join(reasons))

    @property
    def name(self) -> str:
        return self.backends[self.active].name

    def _call(self, fn: Callable[[NluBackend], object]):
        last: Exception | None = None
        for i in range(self.active, len(self.backends)):
            b = self.backends[i]
            try:
                out = fn(b)
                if i != self.active:         # the primary timed out (not demoted): this one answered for THIS call only
                    log.warning("%s answered for this call instead of %s", b.name, self.backends[self.active].name)
                return out
            except BackendTimeout as e:      # slow once is not broken: use the next backend for THIS call, keep the primary
                log.warning("backend %s timed out: %s", b.name, e)
                last = e
            except BackendError as e:
                log.warning("backend %s failed: %s", b.name, e)
                last = e
                if i == self.active and i + 1 < len(self.backends):
                    self.active = i + 1      # a hard failure demotes it
        raise BackendError(f"all NLU backends failed; last error: {last}")

    def classify(self, state: State, car: str | None, text: str, last_agent: str = "") -> IntentResult:
        return self._call(lambda b: b.classify(state, car, text, last_agent))      # type: ignore[return-value]

    def complete_json(self, state: State, car: str | None, text: str) -> dict:
        return self._call(lambda b: b.complete_json(state, car, text))  # type: ignore[return-value]

    def lead_in(self, system: str, user: str, max_tokens: int = 36, temperature: float | None = None, deadline_s: float | None = None) -> str:
        return self.backends[self.active].lead_in(system, user, max_tokens, temperature, deadline_s)      # no fallback: a missing lead-in just means the template alone

    def choose(self, system: str, user: str, options: list[str], max_tokens: int = 16, deadline_s: float | None = None) -> str | None:
        """One of `options` when the active backend can constrain its answer (an OpenAI-compatible server); None otherwise (ask for free text)."""
        b = self.backends[self.active]
        return b.choose(system, user, options, max_tokens, deadline_s) if hasattr(b, "choose") else None

    def lead_in_stream(self, system: str, user: str, max_tokens: int = 36, temperature: float | None = None,
                       deadline_s: float | None = None, prefill: str = "") -> Iterator[str]:
        if prefill:
            return self.backends[self.active].lead_in_stream(system, user, max_tokens, temperature, deadline_s, prefill=prefill)
        return self.backends[self.active].lead_in_stream(system, user, max_tokens, temperature, deadline_s)

    def answer(self, question: str, context: str, max_sentences: int = 2) -> Iterator[str]:
        return self.backends[self.active].answer(question, context, max_sentences)
