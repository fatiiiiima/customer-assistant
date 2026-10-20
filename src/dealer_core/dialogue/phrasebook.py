"""Approved wording variants for low-risk replies (greetings, clarifying questions, "no match", off-topic).

How it works
  * Every situation has ONE base template (the wording the demo was verified with) and a list of placeholders.
  * `config/phrase_variants.yaml` holds extra wordings written offline with the local model. Each has a status. Only entries a human
    has marked `approved` are ever used, and only when `dialogue.variants: true` in the config (default: off).
  * At run time `Phrasebook.say()` picks one of [base + approved variants] with a seeded random generator, never the same one twice
    in a row for a session. Nothing here calls a model, and the values (counts, filters) are filled in by code afterwards.
  * Every variant is validated (also again when loaded): same placeholders, required phrases verbatim, no new commitment words,
    no flipped negation, the voice rules (1-2 short sentences). An approved variant that fails validation stops start-up.

Not covered on purpose: prices, dates, bookings, read-backs, confirmations, hand-offs. Those stay exact templates.
"""
from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import voice_rules


@dataclass(frozen=True)
class Situation:
    id: str
    base: str
    required: tuple[str, ...] = ()       # phrases every variant must contain verbatim
    note: str = ""                       # what the reply is for (shown to the model that writes variants)

    @property
    def fields(self) -> frozenset[str]:
        return placeholders(self.base)


def placeholders(text: str) -> frozenset[str]:
    return frozenset(name for _, name, _, _ in string.Formatter().parse(text) if name)


# The base wording below is exactly what the engine said before variants existed (tests pin this).
SITUATIONS: dict[str, Situation] = {s.id: s for s in [
    Situation("opening", "Hello, welcome to the AGMC dealership. What kind of car are you looking for?",
              note="the agent's first line of the call"),
    Situation("goodbye", "Thanks for calling. Have a great day!", note="closing the call politely"),
    Situation("hello_again", "Hello again! How can I help?", note="the caller greets again mid-call"),
    Situation("ask_what_car", "What kind of car are you after, and roughly what budget?",
              note="the caller has not said what they want yet"),
    Situation("didnt_catch", "Sorry, could you say that again?", note="the agent did not understand and asks for a repeat"),
    Situation("narrow_make_mileage", "I have {total} {desc}. Do you prefer a particular make, or a maximum mileage?",
              note="too many matches to read out; ask a question to narrow them down"),
    Situation("narrow_budget", "I have {total} {desc}. What's your budget?",
              note="too many matches to read out; ask for the budget"),
    Situation("narrow_year_mileage", "I have {total} {desc}. Is there a year or mileage you'd like to stay within?",
              note="too many matches to read out; ask about year or mileage"),
    Situation("narrow_again", "That's still {total} {desc}. Tell me a make, a model or a mileage limit, "
                              "or say 'just show me' and I'll read out the first three.",
              required=("just show me",), note="still too many matches after a first question; offer to just read them out"),
    Situation("cannot_filter", "I can't search on that, only on make, model, body type, colour, budget, year and mileage. "
                               "What would you like to change?",
              note="the caller asked for something the search cannot filter on"),
    Situation("no_match", "I don't have any {desc} right now. Would you like to change the budget or the type of car?",
              note="nothing in stock matches"),
    Situation("off_topic_ask_car", "Sorry, I can only help with our used cars. What kind of car are you looking for?",
              note="the caller said something unrelated to cars; steer back"),
    Situation("off_topic_this_one", "Sorry, I can only help with our used cars. Is there anything else you'd like to know about this one?",
              note="the caller said something unrelated while a car is being discussed; steer back"),
]}

# Words that would add a promise or a commercial claim the templates never made.
FORBIDDEN = re.compile(r"\b(?:guarantee\w*|promise\w*|assure\w*|make sure|definitely|certainly|absolutely|discount\w*|offers?|deals?|"
                       r"free|warrant\w*|financ\w*|lowest|cheapest|best price|special)\b", re.I)
_NEGATION = re.compile(r"(?:\bnot\b|\bno\b|\bnever\b|\bcannot\b|n't|n’t|\bunable\b)", re.I)
MAX_EXTRA_WORDS = 14                      # a variant may not be much longer than the base wording


def check_variant(sit: Situation, text: str) -> list[str]:
    """Why `text` is NOT a safe variant of `sit` (empty list = it is)."""
    problems: list[str] = []
    text = text.strip()
    if not text:
        return ["empty"]
    if placeholders(text) != sit.fields:
        problems.append(f"placeholders {sorted(placeholders(text))} != {sorted(sit.fields)}")
    for phrase in sit.required:
        if phrase not in text:
            problems.append(f"missing required phrase '{phrase}'")
    if FORBIDDEN.search(text) and not FORBIDDEN.search(sit.base):
        problems.append(f"new commitment/commercial word '{FORBIDDEN.search(text).group(0)}'")
    if bool(_NEGATION.search(text)) != bool(_NEGATION.search(sit.base)):
        problems.append("negation differs from the base wording")
    if len(text.split()) > len(sit.base.split()) + MAX_EXTRA_WORDS:
        problems.append("much longer than the base wording")
    try:
        sample = text.format_map(_Blank())
    except (KeyError, ValueError, IndexError):
        return problems + ["malformed placeholder braces"]
    problems += [f"voice rule: {v}" for v in voice_rules.check(sample)]
    if text == sit.base:
        problems.append("identical to the base wording")
    return problems


class _Blank(dict):
    def __missing__(self, key: str) -> str:       # placeholders filled with a neutral word, only to run the voice-rule check
        return "item"


def load_variants(path: Path, statuses: tuple[str, ...] = ("approved",)) -> dict[str, list[str]]:
    """{situation id: [text, ...]} for entries whose status is in `statuses`. Raises ValueError for an invalid entry."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, list[str]] = {}
    for sid, entries in data.items():
        if sid not in SITUATIONS:
            raise ValueError(f"{path.name}: unknown situation '{sid}'")
        for e in entries or []:
            if e.get("status") not in statuses:
                continue
            bad = check_variant(SITUATIONS[sid], e["text"])
            if bad:
                raise ValueError(f"{path.name}: variant for '{sid}' is not valid ({'; '.join(bad)}): {e['text']!r}")
            out.setdefault(sid, []).append(e["text"].strip())
    return out


class Phrasebook:
    def __init__(self, variants: dict[str, list[str]] | None = None) -> None:
        self.variants = variants or {}

    @classmethod
    def from_file(cls, path: Path) -> "Phrasebook":
        return cls(load_variants(path))

    def say(self, rng: random.Random, last: dict[str, int], sid: str, **values: object) -> str:
        """The base wording or an approved variant (never the same one twice in a row), with the values filled in."""
        sit = SITUATIONS[sid]
        options = [sit.base, *self.variants.get(sid, [])]
        choices = [i for i in range(len(options)) if i != last.get(sid)] or [0]
        i = rng.choice(choices)
        last[sid] = i
        return options[i].format(**values)
