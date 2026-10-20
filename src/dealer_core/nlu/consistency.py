"""Deterministic consistency checks between the LLM's intent and what the caller's words actually contain.

A small model occasionally picks an intent that contradicts the utterance. These checks follow from the DEFINITIONS of the intents
(not from any particular test sentence) and only move an intent to a sibling that is valid in the same states.
"""
from __future__ import annotations

import re

from ..dialogue.states import STATE_SPEC, Intent, State
from . import extractors as ex


# Words that can plausibly mean "talk about the price". A price question with none of these (and no amount) cannot be one.
_PRICE_WORDS = re.compile(r"price|cost|cheap|lower|lowest|less|best|deal|discount|offer|final|negotiat|bargain|budget|afford|expens|"
                          r"much|pay|down|reduc|drop|move|flexib|room|wiggle|haggle|dirham|aed|worth|steep|high|too|bit|sell|take it|"
                          r"how about|what about|can you do|could you do|any chance")


def repair(intent: Intent, state: State, text: str) -> tuple[Intent, str | None]:
    """Returns (intent, reason). `reason` is None when nothing was changed."""
    allowed = STATE_SPEC[state][0] if state in STATE_SPEC else []
    # "ask_lowest_price" means asking for the best/lowest price WITHOUT naming one. If the caller names an amount it is an offer.
    if intent is Intent.ASK_LOWEST_PRICE and Intent.MAKE_OFFER in allowed and ex.offer_amount(text) is not None:
        return Intent.MAKE_OFFER, "named an amount, so it is an offer, not a lowest-price question"
    # A price intent changes the negotiation (it can spend a round and speak a "final" price), so it needs some price wording.
    # "bently" on its own is not one. Fall back to out_of_scope, which only asks the caller to say it again.
    if intent in (Intent.ASK_LOWEST_PRICE, Intent.MAKE_OFFER) and ex.offer_amount(text) is None and not _PRICE_WORDS.search(text.lower()):
        return Intent.OUT_OF_SCOPE, "no amount and no price wording, so it is not a price question"
    return intent, None
