"""Slot extractors: deterministic code that turns the caller's words into values. The LLM only chooses the intent; every slot
value comes from here, so it is grounded in the utterance by construction.

The make/model gazetteer is built from the live inventory (`InventoryRepository.vocabulary()`), so it always matches what is
for sale. Fuzzy matching (rapidfuzz) tolerates speech-recognition slips such as "toyode".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from ..models import CarView
from .numbers import normalise, number_spans
from .datetime_parse import When, parse_when  # noqa: F401  (re-exported)
from ..tools.phones import normalize_phone

MAKE_ALIASES = {"mercedes benz": "mercedes", "benz": "mercedes", "merc": "mercedes", "chevy": "chevrolet", "landrover": "land rover"}
BODY = {"suv": "suv", "suvs": "suv", "crossover": "suv", "4x4": "suv", "sedan": "sedan", "sedans": "sedan", "saloon": "sedan",
        "hatchback": "hatchback", "hatch": "hatchback", "coupe": "coupe", "pickup": "pickup", "pickups": "pickup",
        "truck": "pickup", "trucks": "pickup"}
KM_TOKENS = {"km", "kms", "kilometre", "kilometres", "kilometer", "kilometers", "miles"}
_OTHERS = re.compile(r"\b(other|others|different|else|another|more|alternatives?|something like|not that|not those)\b")


def prenormalise(text: str) -> str:
    """'150k' -> '150000', '1.5 million' -> '1500000' (digits only; words are handled by the number parser)."""
    t = re.sub(r"(\d+(?:\.\d+)?)\s*k\b(?!m)", lambda m: str(int(float(m[1]) * 1000)), text, flags=re.I)
    return re.sub(r"(\d+(?:\.\d+)?)\s*million\b", lambda m: str(int(float(m[1]) * 1_000_000)), t, flags=re.I)


def tokens_of(text: str) -> list[str]:
    t = prenormalise(text.lower().replace("’", "'"))
    t = re.sub(r"\b(\w+)n't\b", r"\1 not", t)
    return normalise(t).split()


# ---------------------------------------------------------------------------------------------------- vehicles
def find_vehicle(tokens: list[str], vocab: dict[str, list[str]]) -> dict:
    """{'make': ..., 'model': ...} (either may be absent) found in the tokens, with fuzzy matching on longer names."""
    out: dict = {}
    text = f" {' '.join(tokens)} "
    for alias, make in MAKE_ALIASES.items():
        if f" {alias} " in text and make in vocab:
            out["make"] = make
    grams = [" ".join(tokens[i:i + n]) for n in (1, 2, 3) for i in range(len(tokens) - n + 1)]
    makes = list(vocab)
    if "make" not in out:
        exact = [m for m in makes if f" {m} " in text]
        if exact:
            out["make"] = max(exact, key=len)
        else:
            scored = [(g, process.extractOne(g, makes, scorer=fuzz.ratio, score_cutoff=80)) for g in grams if len(g) >= 4]
            scored = [(g, m) for g, m in scored if m]
            if scored:
                out["make"] = max(scored, key=lambda x: x[1][1])[1][0]
    flat = [(model, make) for make, models in vocab.items() for model in models]
    hits = [(m, mk) for m, mk in flat if f" {m} " in text and (len(m) > 2 or "make" in out)]
    if not hits:
        long_models = [(m, mk) for m, mk in flat if len(m) >= 6]
        for g in grams:
            if len(g) >= 6 and long_models:
                r = process.extractOne(g, [m for m, _ in long_models], scorer=fuzz.ratio, score_cutoff=88)
                if r:
                    hits.append((r[0], dict(long_models)[r[0]]))
                    break
    if hits:
        model, make = max(hits, key=lambda h: len(h[0]))
        if "make" not in out or out["make"] == make:
            out["make"], out["model"] = make, model
    return out


def search_filters(text: str, vocab: dict[str, list[str]]) -> dict:
    """Filters the caller asked for: make, model, body_type, year_min, max_price, max_km (only those actually said)."""
    toks = tokens_of(text)
    f = find_vehicle(toks, vocab)
    for t in toks:
        if t in BODY:
            f["body_type"] = BODY[t]
    for val, i, j in number_spans(toks):
        n = int(val) if val.isdigit() else 0
        nxt = toks[j] if j < len(toks) else ""
        if nxt in KM_TOKENS:
            f["max_km"] = n
        elif len(val) == 4 and 1990 <= n <= 2030:
            f["year_min"] = n
        elif n >= 5000 and not val.startswith("0"):
            f["max_price"] = n
    return f


def wants_others(text: str) -> bool:
    return bool(_OTHERS.search(" ".join(tokens_of(text))))


def offer_amount(text: str) -> int | None:
    """The price the caller names (AED). Kilometres, years and phone numbers are excluded. If several, the last one wins."""
    toks = tokens_of(text)
    amounts = [int(v) for v, i, j in number_spans(toks)
               if v.isdigit() and not v.startswith("0") and int(v) >= 1000 and not (len(v) == 4 and 1990 <= int(v) <= 2030)
               and (toks[j] if j < len(toks) else "") not in KM_TOKENS]
    return amounts[-1] if amounts else None


def selection(text: str, shown: list[CarView]) -> str | None:
    """Which of the shown cars the caller means: ordinal ('the second one'), name ('the Patrol'), or 'the cheapest'."""
    if not shown:
        return None
    toks = tokens_of(text)
    joined = " ".join(toks)
    ordinal = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
    m = re.search(r"\b(first|1st|second|2nd|third|3rd)\b", joined)
    if m and ordinal[m[1]] < len(shown):
        return shown[ordinal[m[1]]].stock_id
    if re.search(r"\blast\b", joined):
        return shown[-1].stock_id
    if re.search(r"\b(cheapest|cheaper|lowest price)\b", joined):
        return min(shown, key=lambda c: c.price).stock_id
    if re.search(r"\b(newest|most recent)\b", joined):
        return max(shown, key=lambda c: c.year).stock_id
    if re.search(r"\b(lowest mileage|fewest kilomet\w+)\b", joined):
        return min(shown, key=lambda c: c.km).stock_id
    vocab: dict[str, list[str]] = {}
    for c in shown:
        vocab.setdefault(c.make.lower(), []).append(c.model.lower())
    v = find_vehicle(toks, vocab)
    matches = [c for c in shown if (v.get("model") and c.model.lower() == v["model"])
               or (not v.get("model") and v.get("make") and c.make.lower() == v["make"])]
    return matches[0].stock_id if len(matches) == 1 else None


# ---------------------------------------------------------------------------------------------------- car questions
ATTRIBUTES = [
    ("km", r"kilomet|\bkm\b|mileage|miles|driven|how far"), ("color", r"colou?r"),
    ("year", r"which year|what year|model year|how old|\byear\b"), ("price", r"how much|price|cost|asking"),
    ("fuel", r"petrol|diesel|hybrid|fuel|electric"), ("transmission", r"automatic|manual|gearbox|transmission"),
    ("spec", r"\bgcc\b|\bspec\b|specs\b|american|japan|import"), ("service", r"service history|serviced|service record|maintenance"),
    ("accident", r"accident|damage|repaint|flood|crash"), ("branch", r"branch|where is|location|which showroom"),
    ("availability", r"still available|is it available|sold|reserved"),
]


def attributes_asked(text: str) -> list[str]:
    t = " ".join(tokens_of(text))
    return [k for k, rx in ATTRIBUTES if re.search(rx, t)]


# ---------------------------------------------------------------------------------------------------- contact details
_NAME_CUE = re.compile(r"(?:my name is|this is|i am|i'm|it is|it's|its|name's|call me|name:)\s+([A-Za-z][A-Za-z' ]*?)"
                       r"(?=\s+(?:and|my|phone|number|mobile|call|reach|you can)\b|[,.;:\d]|\s+(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)\b|$)",
                       re.I)
_STOP = {"yes", "no", "ok", "okay", "my", "the", "hello", "hi", "sure", "thanks", "thank", "please", "it", "its", "it's", "i", "i'm",
        "im", "am", "is", "this", "that", "that's", "number", "phone", "mobile", "and", "so", "well", "um", "uh"}


@dataclass(frozen=True)
class Contact:
    name: str | None
    phone: str | None            # E.164 when valid
    heard_digits: str | None     # the digits heard, even when they do not form a valid UAE number (to ask again)


def _title(name: str) -> str:
    return " ".join(w.capitalize() if w.lower() != "al" else "Al" for w in name.split())


def contact(text: str) -> Contact:
    digits = None
    for val, _, _ in number_spans(tokens_of(text)):
        if val.isdigit() and 8 <= len(val) <= 13:
            digits = val
    phone = normalize_phone(digits) if digits else None
    name = None
    m = _NAME_CUE.search(text)
    if m:
        name = m[1].strip()
    else:                                   # "Mohammed Al Mansoori, zero five zero ..." : the leading words before the number
        lead = re.split(r"[,.;:\d]| (?:zero|oh|one|two|three|four|five|six|seven|eight|nine)\b| and | my ", text, maxsplit=1, flags=re.I)[0]
        words = [w for w in lead.strip().split() if re.fullmatch(r"[A-Za-z']+", w)]
        if 1 <= len(words) <= 4 and not any(w.lower() in _STOP for w in words):
            name = " ".join(words)
    if name:
        name = " ".join(w for w in name.split() if w.lower() not in {"and", "my"})
        name = _title(name) if name else None
    return Contact(name=name, phone=phone, heard_digits=digits)
