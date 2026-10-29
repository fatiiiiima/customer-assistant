"""Fact fidelity for a model-written reply (ARCH_PLAN Phase 3): every number, year, price, colour, car name and commercial term in it must come from the facts given
for THIS turn or from the caller's own words. Fail closed: a sentence with a leak is not spoken and the code's own wording is.

`dialogue/writer.py: verify_sentence` already rejects invented numbers, names, colours and claims, but it accepts anything said earlier in the call as known, so a
figure of another car mentioned three turns ago passes. This check does not look at the history, and it knows the stock's makes and models in any case (lower case
too), so it is the stricter of the two. It runs in addition, only where the model writes the whole reply (`dialogue.model_speaks`).
"""
from __future__ import annotations

import re

from ..dialogue.draft import numbers
from ..nlu.extractors import COLORS

# Terms that promise something commercial. One may be said only when this turn's facts say it ("comes with an AGMC warranty of 6 months" may be repeated; "you can return
# it within a week" may not be invented). The writer's own check caught 16 of 17 such sentences in tests/unit/test_model_speaks.py; it missed the return policy.
COMMERCIAL = re.compile(r"\b(?:return\w*|refund\w*|money back|warrant\w*|guarant\w*|financ\w*|loan\w*|instal+ments?|monthly|deliver\w*|insur\w*|deposit\w*|trade[- ]?in|"
                        r"part[- ]exchange|discount\w*|negotiab\w*|free|complimentary|included|reserve\w*|hold it)\b", re.I)


def _words(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9-]+", text)}


def car_names(vocabulary: dict[str, list[str]]) -> set[str]:
    """Every make and model word in stock, lower case ("toyota", "rav4", "land", "cruiser"), without words too ordinary to be a name on their own."""
    out = set()
    for make, models in vocabulary.items():
        for name in (make, *models):
            out |= {w for w in _words(name) if not w.isdigit() and (len(w) > 2 or (len(w) == 2 and any(c.isdigit() for c in w)))}     # "x5", "q7" count; "3" does not
    return out - {"the", "and", "class", "series", "sport", "line", "plus", "max", "pro", "new", "one"}


def leaks(sentence: str, allowed: str, names: set[str]) -> list[str]:
    """What `sentence` says that `allowed` (the turn's facts, the code's wording and the caller's words) does not: numbers (digits and spoken numbers over ten, years,
    prices), colours, makes and models. Empty when nothing leaks."""
    out: list[str] = []
    extra = numbers(sentence, True) - numbers(allowed)
    if extra:
        out.append(f"number(s) not in this turn's facts: {sorted(extra)[:3]}")
    said, known = _words(sentence), _words(allowed)
    colours = sorted(w for w in said if w in COLORS and w not in known and COLORS[w] not in known)
    if colours:
        out.append(f"colour(s) not in this turn's facts: {colours[:3]}")
    cars = sorted(w for w in said & names if w not in known and w.rstrip("s") not in known)
    if cars:
        out.append(f"car name(s) not in this turn's facts: {cars[:3]}")
    terms = sorted({m.group(0).lower() for m in COMMERCIAL.finditer(sentence)} - {m.group(0).lower() for m in COMMERCIAL.finditer(allowed)})
    if terms:
        out.append(f"commercial term(s) not in this turn's facts: {terms[:3]}")
    return out
