"""Grounding filter: keep a slot only if the utterance actually supports its value.

Small models invent slot values ("offer_amount": 0, a made-up topic). A slot survives only when the value is
present in the customer's words: numbers must be heard (digits or words), strings must be substrings, topics
need a keyword. The filter runs on every LLM slot and is deterministic.
"""
from __future__ import annotations

import re

from .numbers import canon_numbers, normalise

TOPIC_RE = {"warranty": r"warrant", "financing": r"financ|loan|bank|instal", "inspection": r"inspect",
            "trade_in": r"trade|exchange|old car", "documents": r"document|paper|bring|need"}


def ground_slots(utterance: str, slots: dict) -> dict:
    u = " " + normalise(utterance) + " "
    heard = canon_numbers(utterance)
    kept: dict = {}
    for k, v in slots.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            if v != 0 and str(v) in heard:
                kept[k] = v
        elif k == "phone":
            digits = "".join(c for c in str(v) if c.isdigit())
            if digits and digits in heard:
                kept[k] = v
        elif k == "topic":
            if re.search(TOPIC_RE.get(str(v), r"$^"), utterance.lower()):
                kept[k] = v
        else:
            nv = normalise(str(v))
            if nv and f" {nv} " in u:
                kept[k] = v
    return kept
