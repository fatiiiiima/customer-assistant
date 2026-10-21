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


# Words that ask for a better price. Without any of them a price question is just a question.
_BARGAIN = re.compile(r"lowest|best|cheaper|cheap|discount|less|negotiat|deal|final|can you do|could you do|any room|reduce|drop|bottom|"
                      r"flexib|come down|lower|knock|haggle|offer|budget|afford|too (?:much|high|expensive)|expensive|steep|min(?:imum)?|last price|"
                      r"special|bargain|take off|off the|hundred|thousand|\bk\b")


_ANGER = re.compile(r"\b(?:useless|ridiculous|rip ?off|wast(?:e|ing|ed)|stupid|idiot|unacceptable|terrible|awful|fed up|sick of|scam|liar|nonsense|hate|damn|"
                    r"shut up|crap|joke|angry|furious|annoy|frustrat|complain|disgust|rubbish|worst|pathetic|outrageous|disappoint|upset|"
                    r"not happy|unhappy|fuck|shit|wtf|are you serious|what the|sucks|garbage|horrible|pointless|bullshit|lying|cheat|fraud|robbery)")


_OFFER_WORDS = re.compile(r"can you (?:do|sell)|could you (?:do|sell)|would you (?:do|take|sell)|will you (?:take|do|sell)|i can (?:do|pay)|"
                          r"i could (?:do|pay)|how about|what about|take|offer|pay|sell")


def repair(intent: Intent, state: State, text: str, has_car: bool = False) -> tuple[Intent, str | None]:
    """Returns (intent, reason). `reason` is None when nothing was changed."""
    allowed = STATE_SPEC[state][0] if state in STATE_SPEC else []
    # "angry" ends the call with a hand-off, so it needs a sign of anger in the words. "its so hot today" is small talk, not a complaint.
    if intent is Intent.ANGRY and Intent.OUT_OF_SCOPE in allowed and not _ANGER.search(text.lower()):
        return Intent.OUT_OF_SCOPE, "no sign of anger in the words, so it is not an angry customer"
    # An amount said with offer wording about the car being discussed is an offer, not a search ("can you sell the ford at 100,000").
    if intent is Intent.SEARCH_CARS and has_car and Intent.MAKE_OFFER in allowed and ex.offer_amount(text) is not None \
            and _OFFER_WORDS.search(text.lower()):
        return Intent.MAKE_OFFER, "an amount with offer wording about the car in focus is an offer"
    # "ask_lowest_price" means asking for the best/lowest price WITHOUT naming one. If the caller names an amount it is an offer.
    if intent is Intent.ASK_LOWEST_PRICE and Intent.MAKE_OFFER in allowed and ex.offer_amount(text) is not None:
        return Intent.MAKE_OFFER, "named an amount, so it is an offer, not a lowest-price question"
    # A price question with no bargaining in it ("how much is it", "ok and the price?") asks the asking price: answer with the price,
    # do not open a negotiation. The model often labels these "lowest price".
    if intent is Intent.ASK_LOWEST_PRICE and Intent.ASK_CAR_INFO in allowed and ex.offer_amount(text) is None and not _BARGAIN.search(text.lower()) and re.search(r"price|cost|how much|asking", text.lower()):
        return Intent.ASK_CAR_INFO, "no bargaining in it: a plain price question is answered with the asking price"
    # A price intent changes the negotiation (it can spend a round and speak a "final" price), so it needs some price wording.
    # "bently" on its own is not one. Fall back to out_of_scope, which only asks the caller to say it again.
    if intent in (Intent.ASK_LOWEST_PRICE, Intent.MAKE_OFFER) and ex.offer_amount(text) is None and not _PRICE_WORDS.search(text.lower()):
        return Intent.OUT_OF_SCOPE, "no amount and no price wording, so it is not a price question"
    return intent, None
