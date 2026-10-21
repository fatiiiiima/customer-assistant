"""Voice rules for every spoken reply: 1-2 short sentences, no markdown, no bullets, no URLs, at most 3 cars named.

The engine runs `check()` on every reply and records violations in the debug trace; the tests assert that no scenario produces one.
"""
from __future__ import annotations

import re

MAX_SENTENCES = 2
MAX_WORDS = 55            # read-backs (name, phone, date, time) are the longest legitimate replies
MAX_CARS = 3
_MARKDOWN = re.compile(r"(\*\*|__|^#{1,6}\s|`|^\s*[-*•]\s|\[[^\]]+\]\([^)]+\)|^\s*\d+\.\s)", re.M)
_URL = re.compile(r"(https?://|www\.|\b[\w-]+\.(?:com|net|org|ae|io)\b)", re.I)
_CAR = re.compile(r"\b(?:19|20)\d{2} [A-Z][\w-]+(?: [A-Z][\w-]+)*")


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s]


def check(reply: str, max_cars: int = MAX_CARS, max_sentences: int = MAX_SENTENCES, max_words: int = MAX_WORDS) -> list[str]:
    """Violations of the voice rules (empty list = fine)."""
    out: list[str] = []
    n = len(sentences(reply))
    if n == 0:
        out.append("empty reply")
    if n > max_sentences:
        out.append(f"{n} sentences (max {max_sentences})")
    if len(reply.split()) > max_words:
        out.append(f"{len(reply.split())} words (max {max_words})")
    if _MARKDOWN.search(reply):
        out.append("markdown or bullets")
    if _URL.search(reply):
        out.append("URL")
    if len(_CAR.findall(reply)) > max_cars:
        out.append(f"more than {max_cars} cars named")
    if "\n" in reply.strip():
        out.append("line break")
    return out
