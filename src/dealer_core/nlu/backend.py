"""NLU backend interface. Backends are chosen by config (`llm.backend`, `llm.text_backend`, `llm.fallbacks`)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

from ..config import BackendCfg
from ..dialogue.states import Intent, State


class BackendError(RuntimeError):
    """A backend call failed (device missing, timeout, malformed output). The chain moves to the next backend."""


class BackendTimeout(BackendError):
    """A call exceeded its time budget. The backend may be fine (just slow once): the chain falls back for THIS call only."""


@dataclass(frozen=True)
class BackendHealth:
    ok: bool
    detail: str


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    latency_ms: float
    backend: str
    raw: str


def resolve_prompt_mode(cfg: BackendCfg) -> str:
    """'state' or 'static'. On the iGPU a prefix-cache miss costs ~160 ms so a short per-state prompt wins; on CPU
    (OpenVINO or Ollama) a miss costs 1.2 s+, so one static prompt keeps the prefix cached (Phase 0b measurements)."""
    if cfg.prompt_mode != "auto":
        return cfg.prompt_mode
    return "state" if cfg.runtime == "openvino" and cfg.device in ("GPU", "NPU") else "static"


class NluBackend(ABC):
    name: str

    @abstractmethod
    def health(self) -> BackendHealth:
        """Cheap-but-real check used at startup: is the device/server present and the model loadable?"""

    def warm_up(self) -> None:
        """Optional: pre-compile grammars / load weights so the first real caller does not pay the cold-start cost."""

    def lead_in(self, system: str, user: str, max_tokens: int = 36, temperature: float | None = None, deadline_s: float | None = None) -> str:
        """Free text, a few words: the conversational lead-in for hybrid replies (verified by the caller before use)."""
        raise BackendError(f"{self.name} cannot write lead-ins")

    def lead_in_stream(self, system: str, user: str, max_tokens: int = 36, temperature: float | None = None,
                       deadline_s: float | None = None, prefill: str = "") -> Iterator[str]:
        """Free text as it is produced (pieces of a few characters). Backends that cannot stream give the whole text at once.
        `prefill`: the reply already begun by code. A backend that cannot continue a started reply is told what was said instead."""
        if prefill:
            user = f"{user}\n(You have already said: \"{prefill.strip()}\". Write only what comes next.)"
        yield self.lead_in(system, user, max_tokens, temperature, deadline_s)

    def structured(self, system: str, user: str, schema: dict, max_tokens: int = 120, deadline_s: float | None = None) -> str | None:
        """ONE JSON object, as text, that follows `schema` (the single-call turn: agent/turn_schema.py). Raises BackendTimeout past `deadline_s`. None when this backend cannot
        hold its output to a schema, or its server refused to: the caller then uses the two-call path."""
        return None

    @abstractmethod
    def classify(self, state: State, car: str | None, text: str, last_agent: str = "") -> IntentResult:
        """Production call: grammar-constrained, intent only (~6 tokens). Slots are extracted by code afterwards."""

    @abstractmethod
    def complete_json(self, state: State, car: str | None, text: str) -> dict:
        """Intent + slots as JSON. Used for accuracy comparisons between backends, not on the live path."""

    @abstractmethod
    def answer(self, question: str, context: str, max_sentences: int = 2) -> Iterator[str]:
        """Free-form reply for open questions, yielded one sentence at a time so TTS can start on the first."""
