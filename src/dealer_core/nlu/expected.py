"""Read a short reply against the question the agent just asked.

"Any budget?" means nothing alone and everything after "Could you tell me your budget?". The model sees the last agent line too, but its choice
for such short replies varies with the surrounding context (it has answered "Any budget?" with "connect you with a colleague"). The question
the agent asked is known exactly, so the common answers to it are decided here, by code, before the model is asked.

Narrowing questions (budget, make, colour, mileage, "what kind of car") are answered by: nothing in particular ("any", "doesn't matter"), a bare
amount, or the thing asked for. All of these are a search; the dialogue engine then lists what matches.
"""
from __future__ import annotations

import re

from . import extractors as ex

# The last question the agent asked, by what it was about.
KINDS: list[tuple[str, re.Pattern]] = [
    ("budget", re.compile(r"\b(?:budget|spend|spending|price range|how much (?:are|do) you)\b")),
    ("mileage", re.compile(r"\b(?:mileage|kilomet(?:re|er)s)\b")),
    ("colour", re.compile(r"\bcolou?r\b")),
    ("make", re.compile(r"\b(?:make|brand)\b")),
    ("kind", re.compile(r"\bwhat (?:kind|type|sort) of car\b|\bwhat are you looking for\b|\blooking for today\b")),
    ("which_one", re.compile(r"\bwhich one\b|\bhear more about\b|\bthe first, the second\b")),
]
NARROWING = {"budget", "mileage", "colour", "make", "kind"}

_NOTHING_IN_PARTICULAR = re.compile(
    r"^(?:(?:um+|uh+|well|oh|just|i|its|it's|it is|i have|ive got|i've got|i dont have|i do not have)\s+)*"
    r"(?:any|anything|whatever|either|no\s+(?:particular|specific)?\s*(?:preference|budget|limit|make|colou?r|mileage)|"
    r"(?:any|no)\s+(?:budget|limit|price|make|colou?r|mileage|preference|particular|one|range|style|type|size|brand|kind|model)|not\s+(?:really\s+)?(?:fussed|bothered|fussy|sure)|"
    r"(?:it|that)\s+(?:does\s*not|doesnt|doesn't)\s+matter|does\s*not\s+matter|i\s+(?:do\s*not|dont|don't)\s+mind|flexible|open|surprise me)"
    r"(?:\s+(?:(?:or|of|and|any)\s+)*(?:budget|limit|price|make|colou?r|mileage|preference|one|range|style|type|size|brand|kind|model|is fine|will do|works|for me|really))*$")


def last_question(last_agent: str) -> str:
    """The final question sentence of the agent's last turn ('' if it did not end on a question)."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", last_agent.strip()) if p.strip()]
    return parts[-1] if parts and parts[-1].endswith("?") else ""


def question_kind(last_agent: str) -> str | None:
    q = last_question(last_agent).lower()
    if not q:
        return None
    return next((name for name, rx in KINDS if rx.search(q)), None)


def nothing_in_particular(text: str) -> bool:
    return bool(_NOTHING_IN_PARTICULAR.match(" ".join(ex.tokens_of(text))))


def answers_narrowing(kind: str, text: str, vocab: dict) -> str | None:
    """A rule name if `text` plainly answers the narrowing question `kind`, else None (the model decides)."""
    toks = ex.tokens_of(text)
    if not toks or len(toks) > 8:
        return None
    if nothing_in_particular(text) or ex.wants_listing(text):
        return "no-preference"
    f = ex.search_filters(text, vocab)
    wanted = {"budget": "max_price", "mileage": "max_km", "colour": "color", "make": "make"}.get(kind)
    if f and (wanted is None or wanted in f):
        return f"answer-{kind}"
    return None
