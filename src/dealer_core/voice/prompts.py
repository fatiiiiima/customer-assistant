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
CONTACT = "my name is, my number is, zero five zero, double five, two"
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
    if state is State.CONTACT:
        return CONTACT + ", " + ", ".join(groups.get("common names", [])[:10])
    if state is State.CONFIRM and not _ASKS_WHEN.search(q):
        return CONFIRM
    if state is State.SLOT or _ASKS_WHEN.search(q):
        return DAY_TIME
    stock = [x for mk, models in (vocab or {}).items() for x in (mk.title(), *[m.title() if not any(ch.isdigit() for ch in m) else m.upper() for m in models])]
    seen: set[str] = set()
    cars = [x for x in stock + groups.get("makes", [])[:25] + groups.get("models", [])[:25] if not (x.lower() in seen or seen.add(x.lower()))][:60]   # what we actually sell comes first            # no place names: Whisper then hears "Abu Dhabi" in "hello"
    return ", ".join(cars + ["test drive", "tomorrow", "today"])
