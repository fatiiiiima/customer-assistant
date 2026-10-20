"""Deterministic consistency checks between the LLM's intent and what the caller's words actually contain.

A small model occasionally picks an intent that contradicts the utterance. These checks follow from the DEFINITIONS of the intents
(not from any particular test sentence) and only move an intent to a sibling that is valid in the same states.
"""
from __future__ import annotations

from ..dialogue.states import STATE_SPEC, Intent, State
from . import extractors as ex


def repair(intent: Intent, state: State, text: str) -> tuple[Intent, str | None]:
    """Returns (intent, reason). `reason` is None when nothing was changed."""
    allowed = STATE_SPEC[state][0] if state in STATE_SPEC else []
    # "ask_lowest_price" means asking for the best/lowest price WITHOUT naming one. If the caller names an amount it is an offer.
    if intent is Intent.ASK_LOWEST_PRICE and Intent.MAKE_OFFER in allowed and ex.offer_amount(text) is not None:
        return Intent.MAKE_OFFER, "named an amount, so it is an offer, not a lowest-price question"
    return intent, None
