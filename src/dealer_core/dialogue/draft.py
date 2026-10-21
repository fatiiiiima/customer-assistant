"""Draft mode (EXPERIMENT): the model writes the WHOLE spoken reply for low-risk turns, from the facts the code just produced.

Flow for an eligible turn (search, car question, off-topic; never price, booking, confirmation, hand-off):
  1. the model picks the intent (grammar-constrained, as always)
  2. code runs the handler and produces the CORE reply (facts, numbers, next step) exactly as before
  3. the model rewrites the CORE into a natural reply for what the caller just said   <- this module
  4. `verify_draft` checks the draft against the CORE; on any failure the CORE is spoken unchanged

The model is never asked for a fact: everything it may say is in the CORE or in the caller's words. Verification is mechanical:
  * every number in the draft is in the CORE or the caller's words, and every number in the CORE is still in the draft
  * every capitalised word (names, makes, models, places) in the draft is in the CORE or the caller's words
  * a question in the CORE is still asked (the next step is not lost); text in 'quotes' in the CORE survives verbatim
  * no new commitment/commercial words, no new first-person promises, no change in negation (a "don't have" cannot become "have")
  * plain characters only; at most 3 sentences and 55 words
"""
from __future__ import annotations

import re
import time

from . import voice_rules
from .leadin import _NEGATION, _PROMISE, LeadInBackend, LeadInOutcome
from .phrasebook import FORBIDDEN

MAX_SENTENCES = 3
MAX_WORDS = 55
DRAFT_TOKENS = 96
DRAFT_TIMEOUT_S = 8.0

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CAPITAL = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][A-Za-z0-9]+\b")
_CHARS = re.compile("^[A-Za-z0-9 ,'" + chr(0x2019) + ".!?%-]+$")
_NEG = re.compile(_NEGATION.pattern + "|" + r"\bno\b|\bnone\b|\bwithout\b", re.I)
_QUOTED = re.compile(r"'([^']{4,})'")


def numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text)}


def capitals(text: str) -> set[str]:
    """Capitalised words that are not just the start of a sentence (names, makes, models, places, acronyms)."""
    out = set()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for w in re.findall(r"[A-Za-z0-9']+", sentence)[1:]:
            if w[:1].isupper() and w != "I":
                out.add(w.lower())
    return out


def verify_draft(draft: str, user_text: str, core: str) -> list[str]:
    """Why `draft` may NOT replace `core` (empty list = it may)."""
    problems: list[str] = []
    draft = draft.strip()
    if not draft:
        return ["empty"]
    if not _CHARS.match(draft):
        problems.append("characters other than plain letters, digits and basic punctuation")
    n_sent = len(voice_rules.sentences(draft))
    if n_sent > MAX_SENTENCES:
        problems.append(f"{n_sent} sentences (max {MAX_SENTENCES})")
    if len(draft.split()) > MAX_WORDS:
        problems.append(f"{len(draft.split())} words (max {MAX_WORDS})")
    allowed_numbers = numbers(core) | numbers(user_text)
    extra = numbers(draft) - allowed_numbers
    if extra:
        problems.append(f"new number(s) {sorted(extra)[:3]}")
    missing = numbers(core) - numbers(draft)
    if missing:
        problems.append(f"dropped number(s) {sorted(missing)[:3]}")
    known_caps = capitals(core) | capitals(user_text) | {w.lower() for w in re.findall(r"[A-Za-z0-9']+", user_text)}
    # a sentence-initial word of the CORE can legitimately appear mid-sentence in a rewrite
    known_caps |= {w.lower() for w in re.findall(r"[A-Za-z0-9']+", core)}
    stems = {w.rstrip("s") for w in known_caps}                              # "Bentleys" is the same name as "Bentley"
    new_caps = {w for w in capitals(draft) if w not in known_caps and w.rstrip("s") not in stems}
    if new_caps:
        problems.append(f"new name(s) {sorted(new_caps)[:3]}")
    if core.rstrip().endswith("?") and not draft.rstrip().endswith("?"):
        problems.append("the next-step question was dropped")
    for q in _QUOTED.findall(core):
        if q not in draft:
            problems.append(f"missing quoted phrase '{q}'")
    if FORBIDDEN.search(draft) and not FORBIDDEN.search(core):
        problems.append(f"new commercial/commitment word '{FORBIDDEN.search(draft).group(0)}'")
    if _PROMISE.search(draft) and not _PROMISE.search(core):
        problems.append("new first-person promise or ability")
    if bool(_NEG.search(draft)) != bool(_NEG.search(core)):
        problems.append("negation differs from the facts")
    if draft.strip().lower() == core.strip().lower():
        problems.append("identical to the template")
    return problems


SYSTEM = """You are the voice of a friendly car dealership phone agent in the UAE. Below are what the CALLER just said and the FACTS the
agent must convey (REPLY). Rewrite REPLY as a natural, warm spoken answer to the caller, in at most 3 short sentences.
Rules: keep every number, price, car name and fact exactly as in REPLY; add no new facts, names, numbers or availability; make no promises
or offers; if REPLY asks the caller a question, end with that same question; if REPLY says something is not available or not possible,
keep that meaning. React to the caller's own words so it sounds like a conversation, not a form. Plain spoken English, no lists, no emojis.
Output only the reply."""


class DraftWriter:
    def __init__(self, backend: LeadInBackend, intents: set[str]) -> None:
        self.backend, self.intents = backend, intents

    def write(self, user_text: str, core: str) -> LeadInOutcome:
        t0 = time.perf_counter()
        try:
            raw = self.backend.lead_in(SYSTEM, f"CALLER: {user_text}\nREPLY: {core}", DRAFT_TOKENS)
        except Exception as e:
            return LeadInOutcome(None, (time.perf_counter() - t0) * 1000, f"backend: {type(e).__name__}")
        ms = (time.perf_counter() - t0) * 1000
        draft = " ".join(raw.strip().split()).strip('"').strip()
        bad = verify_draft(draft, user_text, core)
        return LeadInOutcome(None if bad else draft, ms, "; ".join(bad))
