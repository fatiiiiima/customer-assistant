"""Deterministic NLU fast path, productionised from the Phase 0b prototype (benchmarks/fastpath.py).

PRECISION over coverage: a rule answers only when it is confident, and ONLY for intents whose gate is open. The first
matching rule decides; if that rule's intent is gated closed the engine returns None (defer to the LLM) instead of falling
through to a weaker rule. State-gated: only intents valid in the current state can fire.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import yaml

from ..dialogue.focus import FIND_CUE, about_the_car_in_focus  # noqa: F401
from ..dialogue.states import STATE_CHANGING, STATE_SPEC, Intent, State
from . import extractors as ex
from .datetime_parse import parse_when
from .numbers import number_spans

CONTRACTIONS = [(r"won't", "will not"), (r"can't", "can not"), (r"cannot", "can not"), (r"n't", " not"), (r"let's", "let us"),
                (r"'m\b", " am"), (r"'re\b", " are"), (r"'ll\b", " will"), (r"'ve\b", " have"), (r"'d\b", " would"), (r"'s\b", " is")]

YES = re.compile(r"^(yes|yeah|yep|yup|sure|ok|okay|correct|right|exactly|absolutely|definitely|perfect|fine|deal|done|agreed|good|great|"
                 r"that works|that is right|that is correct|sounds good|let us do it|go ahead|works for me|please do|ok deal|alright)\b")
NO = re.compile(r"^(no|nope|nah|not really|no thanks|i do not think so|that is wrong|that is not right|that is incorrect|i do not want)\b|"
                r"\b(can not make it|cannot make it|not available|not free|does not work|will not work|do not think so)\b")
FILLER = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "thanks", "thank", "you", "that", "is", "it", "its", "right",
          "correct", "works", "for", "me", "sounds", "good", "great", "perfect", "deal", "let", "us", "do", "go", "ahead", "fine",
          "exactly", "absolutely", "definitely", "done", "agreed", "alright", "so", "then", "and", "well", "the", "a"}

RULES: list[tuple[Intent, re.Pattern | None]] = [
    (Intent.ANGRY, re.compile(r"\b(useless|ridiculous|rip ?off|waste (?:of )?my time|stupid|idiots?|unacceptable|terrible|awful|fed up|sick of|"
                              r"scam|liars?|nonsense|hate|damn|shut up|crap|joke)\b")),
    (Intent.CLAIM_AUTHORITY, re.compile(r"\bi (?:am|m) (?:the |a |an )?(?:manager|owner|director|boss|ceo|staff|employee|supervisor)\b|"
                                        r"\b(?:owns?|runs?) (?:the |this )?company\b|\b(?:staff|employee|manager|dealer|trade) (?:price|rate|discount)\b|"
                                        r"\bi work (?:for|at|with) (?:you|the company|this company)\b")),
    (Intent.REQUEST_HUMAN, re.compile(r"\b(?:speak|talk|connect|transfer|put|get)\b.{0,25}\b(?:person|human|agent|someone|somebody|manager|"
                                      r"supervisor|representative|salesperson|salesman|sales|operator)\b|\b(?:real|live) person\b|\bhuman\b|\boperator\b")),
    (Intent.GOODBYE, re.compile(r"\b(?:bye|goodbye|see you|that is all|have a (?:good|nice|great)|talk later)\b")),
    (Intent.CANCEL_BOOKING, re.compile(r"\b(?:cancel|forget (?:it|about)|never ?mind|call it off|scrap)\b")),
    (Intent.RESCHEDULE, re.compile(r"\b(?:reschedule|change (?:it|the|my)|move (?:it|the|my)|another (?:day|time)|different (?:day|time)|instead|postpone)\b")),
    (Intent.ASK_POLICY, None),          # topic keywords, handled in code
    (Intent.SEND_DETAILS, re.compile(r"\b(?:send|mail|email|e mail|whatsapp|share|text|forward)\b.{0,30}\b(?:photos?|pictures?|details?|specs?|link|info|information|brochure)\b|"
                                     r"\b(?:photos?|pictures?|details)\b.{0,20}\b(?:whatsapp|email|message)\b|\bmail me\b")),
    (Intent.BOOK_TEST_DRIVE, re.compile(r"\b(?:test ?drive|try (?:it|the car|out)|drive it|take it for a (?:spin|drive)|see it in person|"
                                        r"come (?:and )?(?:see|view|look|check)|view(?:ing)? the car|visit)\b")),
    (Intent.ASK_LOWEST_PRICE, re.compile(r"\b(?:lowest|best|final|bottom|cheapest|minimum|rock bottom|last)\b.{0,20}\b(?:price|offer|you can (?:do|give|go)|figure)\b|"
                                         r"\bhow low (?:can|could)\b|\bwhat (?:is |would )?(?:your )?(?:best|final|lowest)\b|\bcan you do better\b|\bany discount\b")),
]
TOPICS = {"warranty": r"warrant|guarantee", "financing": r"financ|loan|installment|instalment|monthly payment|bank|mortgage|credit",
          "inspection": r"inspect|mechanic|certif", "trade_in": r"trade ?in|exchange|my (?:old|current) car|sell my car",
          "documents": r"documents?|papers?|what (?:do i|should i) bring|emirates id|requirements"}
INFO_RE = re.compile(r"\b(?:how many (?:km|kilomet\w+|miles|owners|keys)|mileage|colou?r|accident\w*|damage\w*|paint\w*|service history|history|"
                     r"gcc|spec|engine|gearbox|transmission|fuel|petrol|diesel|hybrid|which year|what year|model year|owners?|sunroof|"
                     r"leather|cylinders|horsepower|registration|still available|which branch|where is it)\b")
OFFER_CUE = re.compile(r"\b(?:offer|pay|give you|take|do it for|do|how about|what about|go for|can you do|would you do|cash|for)\b")
SEARCH_CUE = re.compile(r"\b(?:looking for|do you have|have any|any|show me|i want|i need|i am after|after a|interested in|budget|"
                        r"something|cheap|find me|searching)\b")
PRICE_CUE = re.compile(r"\b(?:under|below|less than|up to|max|maximum|budget|around|about|within|not more than|upto)\b")
WANTS_ANOTHER = re.compile(r"\b(?:(?:another|a different|different|other|any other) (?:car|cars|one|ones|option|options|model|models|vehicle)|"
                           r"something (?:else|different)|show me (?:something|others|other)|i want another|looking for another)\b")
ASKING_PRICE = re.compile(r"^(?:how much (?:is|does|would) (?:it|that|this|the \w+)(?: cost)?|how much for (?:it|that|this)|what(?:'s| is) the price(?: of (?:it|that|this))?|"
                         r"what(?:'s| is) it priced at|what does it cost|price)$")
ORDINAL_RE =re.compile(r"\b(?:first|1st|second|2nd|third|3rd|last) one\b")
GREET_RE = re.compile(r"^(hi|hello|hey|good (morning|afternoon|evening)|salam|assalam\w*|marhaba|hiya)\b")


@dataclass(frozen=True)
class RuleHit:
    intent: Intent
    rule: str          # which rule fired, for the debug trace


def expand(text: str) -> str:
    t = text.lower().replace("’", "'")
    for pat, rep in CONTRACTIONS:
        t = re.sub(pat, rep, t)
    return t


class RuleEngine:
    def __init__(self, vocabulary: Callable[[], dict[str, list[str]]], enabled: set[Intent]) -> None:
        self.vocabulary, self.enabled = vocabulary, enabled

    def classify(self, state: State, car: str | None, text: str) -> RuleHit | None:
        """A confident, gate-open, state-valid answer, or None to defer to the LLM."""
        hit = self._first_match(state, car, text)
        if hit is None or hit.intent not in self.enabled:
            return None
        return hit

    def _first_match(self, state: State, car: str | None, text: str) -> RuleHit | None:
        if state not in STATE_SPEC:
            return None
        allowed = set(STATE_SPEC[state][0])
        text = expand(text)
        toks = ex.tokens_of(text)
        norm = " ".join(toks)
        spans = number_spans(toks)

        def ok(intent: Intent, rule: str) -> RuleHit | None:
            return RuleHit(intent, rule) if intent in allowed else None

        if car and state in (State.ANSWERING, State.NEGOTIATING) and ASKING_PRICE.search(re.sub(r"^(?:(?:ok|okay|and|so|then|well|alright)\s+)+", "", norm)):
            return ok(Intent.ASK_CAR_INFO, "asking-price")
        if state in (State.ANSWERING, State.NEGOTIATING) and WANTS_ANOTHER.search(norm):      # "I want another car instead" (not a reschedule)
            return ok(Intent.SEARCH_CARS, "another")
        for intent, pat in RULES:
            if intent is Intent.ASK_POLICY:
                for topic, rx in TOPICS.items():
                    if re.search(rx, norm):
                        return ok(Intent.ASK_POLICY, f"policy:{topic}")
                continue
            if pat and pat.search(norm):
                return ok(intent, intent.value)
        if state is State.SLOT and parse_when(text, _NOW, _TZ).found:
            return ok(Intent.GIVE_SLOT, "when")
        if state is State.CONTACT and ex.contact(text).phone:
            return ok(Intent.GIVE_CONTACT, "phone")
        if about_the_car_in_focus(text, bool(car)):
            return ok(Intent.ASK_CAR_INFO, "about-this-car")       # "what colour is the camry, is it red": a question about THIS car
        vocab = self.vocabulary()
        vehicle = ex.find_vehicle(toks, vocab)
        searchish = bool(vehicle) or any(t in ex.BODY for t in toks) or bool(any(t in ex.COLORS for t in toks) and (FIND_CUE.search(norm) or any(t in ("car", "cars") for t in toks)))
        money = [v for v, i, j in spans if v.isdigit() and int(v) >= 1000 and not v.startswith("0")
                 and not (len(v) == 4 and 1990 <= int(v) <= 2030) and (toks[j] if j < len(toks) else "") not in ex.KM_TOKENS]
        if money and not searchish and (state is State.NEGOTIATING or (state is State.ANSWERING and OFFER_CUE.search(norm))):
            return ok(Intent.MAKE_OFFER, "amount")
        has_km = any((toks[j] if j < len(toks) else "") in ex.KM_TOKENS for _, _, j in spans)
        if searchish or (state in (State.INTEREST, State.GREETING) and spans and (SEARCH_CUE.search(norm) or has_km or PRICE_CUE.search(norm))):
            if ex.search_filters(text, vocab):
                return ok(Intent.SEARCH_CARS, "filters")
        if ex.claims_in_question(text) and (car or state is State.ANSWERING):          # "is it red?": a yes/no question about the car
            return ok(Intent.ASK_CAR_INFO, "claim")
        if INFO_RE.search(norm) and (car or state is State.ANSWERING):
            return ok(Intent.ASK_CAR_INFO, "info")
        if state is State.ANSWERING and ORDINAL_RE.search(norm) and not money:      # "tell me about the second one"
            return ok(Intent.ASK_CAR_INFO, "ordinal")
        words = norm.split()
        if YES.search(norm) and all(w in FILLER for w in words):
            return ok(Intent.CONFIRM_YES, "yes")
        if NO.search(norm) and len(words) <= 10:
            return ok(Intent.CONFIRM_NO, "no")
        if GREET_RE.match(norm) and len(words) <= 4:
            return ok(Intent.GREET, "greeting")
        return None


# `parse_when` needs a clock; the rule only asks "is there a date or time in this?", so any fixed reference works.
from datetime import datetime, timezone  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

_NOW = datetime(2030, 1, 7, 10, 0, tzinfo=timezone.utc)
_TZ = ZoneInfo("Asia/Dubai")


def load_enabled_intents(path) -> set[Intent]:
    """Read config/nlu_gates.yaml. A state-changing intent may only be enabled with recorded evidence of a 95% lower
    bound >= 0.98 (docs/PHASE1_PLAN.md section 5); otherwise loading fails loudly instead of silently trusting a rule."""
    data = yaml.safe_load(open(path, encoding="utf-8")) or {}
    enabled = {Intent(n) for n in data.get("enabled", [])}
    evidence = data.get("evidence") or {}
    for intent in enabled & STATE_CHANGING:
        e = evidence.get(intent.value) or {}
        if float(e.get("lower_bound", 0)) < 0.98:
            raise ValueError(f"gate for state-changing intent {intent.value!r} is open without evidence of a >= 98% lower bound "
                             f"(config says {e or 'none'}); keep it closed until measured")
    return enabled
