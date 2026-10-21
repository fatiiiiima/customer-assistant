"""Hybrid replies: a short model-written LEAD-IN sentence in front of the templated answer.

    caller : "i want another car instead"
    lead-in (model) : "Sure, happy to look at something different."        <- natural, reacts to the caller
    core (template) : "Of course. What kind of car would you like to see instead?"   <- facts, numbers, next step: exact code

The model never sees prices or policy limits and never writes a fact. It is given the caller's words and the core reply that follows,
and asked for ONE short sentence that acknowledges the caller. The sentence is then VERIFIED; on any doubt it is dropped and the turn is
spoken from the template alone, exactly as before. Only low-risk intents are eligible (not price, booking, confirmation, hand-off).

Verification (every rule must pass):
  * one sentence, 2-14 words, a statement (no question: the core asks the question), ends with . or !
  * no digits, currency, or commercial/commitment words (guarantee, promise, discount, offer, deal, free, warranty, financing...)
  * every word is one the caller said, one the core reply says, or one of a fixed list of ordinary conversational words (ALLOWED).
    So it cannot name a car, a make, a place, a feature or a number that is not already in the conversation.
  * no negation except the phrase "no problem"; no first-person ability or promise ("I can", "I will", "let me", "we'll")
  * not the same as the previous lead-in of the call
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Protocol

from .phrasebook import FORBIDDEN

MAX_WORDS = 14
LEAD_TIMEOUT_S = 4.0

# Ordinary words a lead-in may use besides the caller's own and the core reply's. No nouns that carry facts (no cars, prices, places).
ALLOWED = frozenset("""
a an the and or but so then also too just still yet now here there that this those these it its it's that's there's here's
of to for with on in at by from about as like instead rather else other another different new more
i i'd you you're you'd your yours we we're we'd our us
is are was be been being am do does did have has had get got go going come
sure okay ok alright right great good nice lovely fine perfect wonderful excellent of course
oh ah hmm well wow ha haha interesting understood noted cool fair enough thing question area outside budget
yes yeah happy glad pleased delighted keen ready here welcome thanks thank please
help helping look looking see find finding check checking show showing take taking pick picking choose choosing try trying
think thinking say said saying tell telling mean means meant ask asking asked want wanted wanting need needed prefer preferred
understand understood know knew noted hear heard catch caught follow followed sounds sound seems makes make made
something anything everything one ones option options choice choices way ways idea thing things
car cars vehicle vehicles ride
what which who how when where why
can could would should will might may
little bit lot much many some any all both each every few
again once back over out up down off away
first next last another other same
let's lets move moving start starting begin beginning continue
no problem
""".split())

_NEGATION = re.compile(r"(?:\bnot\b|\bnever\b|\bcannot\b|\bnothing\b|\bnone\b|n't|n’t|\bunable\b)", re.I)
_PROMISE = re.compile(r"\b(?:i can|i will|i'll|i shall|we can|we will|we'll|let me|i'm going to|i promise|i'll make|i could|we could)\b", re.I)
_WORD = re.compile(r"[a-z]+(?:['’][a-z]+)?")
_CHARS = re.compile("^[A-Za-z ,'" + chr(0x2019) + ".!-]+$")        # plain letters and basic punctuation only (no dashes, symbols)
_DIGITS_CURRENCY = re.compile(r"\d|dirham|aed|usd|dollar|%")


def words(text: str) -> list[str]:
    return [w.replace("’", "'") for w in _WORD.findall(text.lower())]


def verify(lead: str, user_text: str, core: str, previous: str = "") -> list[str]:
    """Why `lead` may NOT be spoken (empty list = it may)."""
    problems: list[str] = []
    lead = lead.strip()
    ws = words(lead)
    if not (2 <= len(ws) <= MAX_WORDS):
        problems.append(f"{len(ws)} words")
    if "?" in lead or "\n" in lead or not re.search(r"[.!]$", lead) or len(re.findall(r"[.!?]", lead)) != 1:
        problems.append("not exactly one statement")
    if not _CHARS.match(lead) or "--" in lead or " - " in lead:
        problems.append("characters other than plain letters and basic punctuation")
    if _DIGITS_CURRENCY.search(lead.lower()):
        problems.append("number or currency")
    if FORBIDDEN.search(lead):
        problems.append(f"commercial/commitment word '{FORBIDDEN.search(lead).group(0)}'")
    stripped = re.sub(r"\bno problem\b", "", lead, flags=re.I)
    if _NEGATION.search(stripped) or re.search(r"\bno\b", stripped, re.I):
        problems.append("negation")
    if _PROMISE.search(lead):
        problems.append("first-person promise or ability")
    known = ALLOWED | set(words(user_text)) | set(words(core))
    unknown = [w for w in ws if w not in known and w.replace("'s", "") not in known]
    if unknown:
        problems.append(f"words outside the allowed set: {', '.join(unknown[:4])}")
    core_sentences = {re.sub(r"[^a-z' ]", "", x.lower()).strip() for x in re.split(r"(?<=[.!?])\s+", core)}
    if re.sub(r"[^a-z' ]", "", lead.lower()).strip() in core_sentences:
        problems.append("repeats a sentence of the reply")
    if previous and lead.lower() == previous.lower():
        problems.append("same as the previous lead-in")
    return problems


SYSTEM = """You are the voice of a friendly car dealership phone agent in the UAE. A fixed reply (REPLY) will be spoken right after
your sentence. Write ONE very short sentence (3 to 8 words) that simply acknowledges what the caller just said, using their own
words where natural, so the call sounds like a conversation.
Rules: a statement, never a question. Do not answer for the agent, do not describe or praise any car, do not give opinions or
facts, do not mention prices, numbers, places or availability, do not promise or offer anything, do not apologise.
Good examples:
CALLER: i want another car instead -> Sure, let's look at something different.
CALLER: a sedan under 90 thousand -> Got it, a sedan on a budget.
CALLER: is it automatic -> Good question.
CALLER: what's the weather like today -> Ha, that one is outside my area.
CALLER: do you have a land cruiser -> Okay, a Land Cruiser.
CALLER: show me something cheaper -> No problem, something cheaper.
Output only the sentence."""


class LeadInBackend(Protocol):
    def lead_in(self, system: str, user: str, max_tokens: int = 36, temperature: float | None = None) -> str: ...


@dataclass
class LeadInOutcome:
    text: str | None
    ms: float
    why_not: str = ""


class LeadInWriter:
    def __init__(self, backend: LeadInBackend, intents: set[str]) -> None:
        self.backend, self.intents = backend, intents

    def write(self, user_text: str, core: str, previous: str = "") -> LeadInOutcome:
        t0 = time.perf_counter()
        user = f"CALLER: {user_text}\nREPLY: {core}" + (f"\nDo not repeat this earlier sentence: {previous}" if previous else "")
        try:
            raw = self.backend.lead_in(SYSTEM, user)
        except Exception as e:                      # any failure: speak the template alone
            return LeadInOutcome(None, (time.perf_counter() - t0) * 1000, f"backend: {type(e).__name__}")
        ms = (time.perf_counter() - t0) * 1000
        lead = raw.strip().splitlines()[0].strip().strip('"').strip() if raw.strip() else ""
        bad = verify(lead, user_text, core, previous)
        return LeadInOutcome(None if bad else lead, ms, "; ".join(bad))
