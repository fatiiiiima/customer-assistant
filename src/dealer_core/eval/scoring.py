"""Scoring of an NLU prediction against a gold label (intent + slots), with and without the grounding filter."""
from __future__ import annotations

from dataclasses import dataclass

from ..nlu.grounding import ground_slots
from .datasets import Example


@dataclass(frozen=True)
class Score:
    valid: bool
    intent_ok: bool
    exact: bool          # intent and every slot right
    tp: int = 0
    fp: int = 0
    fn: int = 0


def _norm(key: str, value):
    if key == "phone":
        return "".join(c for c in str(value) if c.isdigit())
    return value if isinstance(value, int) else str(value).strip().lower()


def _match(key: str, gold, pred) -> bool:
    if key in ("date", "time", "name", "model", "make"):   # free text: later code normalises these, so be lenient
        return gold in pred or pred in gold
    return gold == pred


def score(ex: Example, pred: dict | None, ground: bool) -> Score:
    if not isinstance(pred, dict) or "intent" not in pred:
        return Score(False, False, False, 0, 0, len(ex.gold_slots))
    slots = pred.get("slots") or {}
    if ground:
        slots = ground_slots(ex.utterance, slots)
    tp = fp = 0
    matched: set[str] = set()
    for k, v in slots.items():
        if k in ex.gold_slots and _match(k, _norm(k, ex.gold_slots[k]), _norm(k, v)):
            tp += 1
            matched.add(k)
        else:
            fp += 1
    fn = sum(1 for k in ex.gold_slots if k not in matched)
    intent_ok = pred["intent"] == ex.gold_intent.value
    return Score(True, intent_ok, intent_ok and fp == 0 and fn == 0, tp, fp, fn)
