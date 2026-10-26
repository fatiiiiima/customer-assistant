"""What to prime the speech recognizer with, by what the agent just asked.

Whisper takes a short text as context and favours those words when the audio is unclear. One fixed list of car brands and first names (the old way)
makes it hear "Ahmed" or "Fiat" in noise and "DeModel" for "tomorrow". The agent knows what it is waiting for, so the list follows the question:
days and times when it asked for a day, names and digits when it asked for a name or number, yes/no when it asked to confirm, cars otherwise.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..dialogue.states import State
from ..nlu.expected import last_question

DAY_TIME = ("tomorrow, today, this afternoon, tonight, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, morning, afternoon, evening, "
            "at ten, at four thirty, test drive")
CONTACT = "050, 055, 056, 052, double five"        # no "my name is": Whisper continues a primed phrase and loops on it ("my name is, my name is, ...")
CONFIRM = "yes, no, that's right, that's correct, change the time, a different day"
_ASKS_WHEN = re.compile(r"(?:\bwhat day|which day|what time|which time|when|works better|suits you|day suits|day and time)\b")


def hotword_groups(path: Path) -> dict[str, list[str]]:
    """config/hotwords.txt split by its '# Heading' lines (makes, models, places, common names)."""
    groups: dict[str, list[str]] = {}
    key = "other"
    if not path.exists():
        return groups
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            key = line.lstrip("# ").strip().lower() or key
        elif line:
            groups.setdefault(key, []).append(line)
    return groups


def stt_prompt(state: State, last_agent: str, groups: dict[str, list[str]], vocab: dict[str, list[str]] | None = None) -> str:
    q = last_question(last_agent).lower()
    if state is State.CONTACT:                    # digits when a number is asked for; no hint for a name (a list of names made Whisper write a list of names)
        return CONTACT if re.search(r"number|mobile|phone|digits", q) else ""
    if state is State.CONFIRM and not _ASKS_WHEN.search(q):
        return CONFIRM
    said = _recent(last_agent)
    if state is State.SLOT or _ASKS_WHEN.search(q):
        return DAY_TIME + said                    # "...tomorrow I can do 9 AM or 2 PM": a bare "two PM" is then heard as that
    stock = [x for mk, models in (vocab or {}).items() for x in (mk.title(), *[m.title() if not any(ch.isdigit() for ch in m) else m.upper() for m in models])]
    seen: set[str] = set()
    cars = [x for x in stock + groups.get("makes", [])[:25] + groups.get("models", [])[:25] if not (x.lower() in seen or seen.add(x.lower()))][:60]   # what we actually sell comes first            # no place names: Whisper then hears "Abu Dhabi" in "hello"
    return ", ".join(cars[:45] + ["white, black, grey, silver, red, blue", "the second one", "test drive", "tomorrow", "today"]) + said


def _recent(last_agent: str, limit: int = 200) -> str:
    """What the agent just said, at the end of the hint: Whisper reads the hint as the conversation so far, so the cars just named win.
    Measured (48 synthetic clips after a Toyota list): 44/48 with the stock names alone, 47/48 with this added; "the RAV4" stopped being "RAV E4"."""
    text = last_agent.strip()
    if not text:
        return ""
    if len(text) > limit:
        text = text[-limit:].split(" ", 1)[-1]
    return ". " + text
