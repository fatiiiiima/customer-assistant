"""NLU pipeline: language check -> rules (gate-open intents only) -> LLM intent call (grammar-constrained) -> slots by code.

`understand()` returns the intent and HOW it was reached, for the debug trace. Slots are extracted later by the dialogue
engine with the extractors, because only the engine knows what it needs for the current state.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Protocol

from ..dialogue.states import Intent, State
from .backend import IntentResult
from .consistency import repair
from .rules import RuleEngine

_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")


class IntentClassifier(Protocol):
    name: str

    def classify(self, state: State, car: str | None, text: str, last_agent: str = "") -> IntentResult: ...


@dataclass(frozen=True)
class Understanding:
    intent: Intent
    via: str                    # "rules" | "llm" | "language"
    rule: str | None = None     # which rule fired (via == "rules")
    backend: str | None = None  # which LLM backend answered (via == "llm")
    ms: float = 0.0
    language: str = "en"
    repaired: str | None = None       # why a consistency check changed the LLM's intent (for the debug trace)


class NluPipeline:
    def __init__(self, classifier: IntentClassifier | None, rules: RuleEngine) -> None:
        self.classifier, self.rules = classifier, rules

    def understand(self, state: State, car: str | None, text: str, last_agent: str = "") -> Understanding:
        t0 = time.perf_counter()
        if _ARABIC.search(text):               # English only for now: the dialogue engine says so and offers a human
            return Understanding(Intent.OUT_OF_SCOPE, "language", language="ar", ms=(time.perf_counter() - t0) * 1000)
        hit = self.rules.classify(state, car, text)
        if hit is not None:
            return Understanding(hit.intent, "rules", rule=hit.rule, ms=(time.perf_counter() - t0) * 1000)
        if self.classifier is None:
            raise RuntimeError("no LLM backend configured and the rules declined this utterance")
        res = self.classifier.classify(state, car, text, last_agent)       # BackendError propagates: the engine hands off to a human
        intent, why = repair(res.intent, state, text, bool(car))
        return Understanding(intent, "llm", backend=res.backend, ms=(time.perf_counter() - t0) * 1000,
                             repaired=f"{res.intent.value} -> {intent.value}: {why}" if why else None)
