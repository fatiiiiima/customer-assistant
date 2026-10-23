"""Slot extractors: deterministic code that turns the caller's words into values. The LLM only chooses the intent; every slot
value comes from here, so it is grounded in the utterance by construction.

The make/model gazetteer is built from the live inventory (`InventoryRepository.vocabulary()`), so it always matches what is
for sale. Fuzzy matching (rapidfuzz) tolerates speech-recognition slips such as "toyode".
"""
from __future__ import annotations

from functools import lru_cache

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from ..models import CarView
from .numbers import normalise, number_spans
from .datetime_parse import When, parse_when  # noqa: F401  (re-exported)
from ..tools.phones import normalize_phone

MAKE_ALIASES = {"b w": "bmw", "b and w": "bmw", "bnw": "bmw", "b m w": "bmw", "bm w": "bmw", "mercedes benz": "mercedes", "benz": "mercedes", "merc": "mercedes", "chevy": "chevrolet", "landrover": "land rover"}
BODY = {"suv": "suv", "suvs": "suv", "crossover": "suv", "4x4": "suv", "sedan": "sedan", "sedans": "sedan", "saloon": "sedan",
        "hatchback": "hatchback", "hatch": "hatchback", "coupe": "coupe", "pickup": "pickup", "pickups": "pickup",
        "truck": "pickup", "trucks": "pickup", "van": "van", "vans": "van", "minivan": "van", "minivans": "van"}
# Makes a caller may ask for that we may not stock. Without these, "do you have a Bentley?" finds no make at all and the agent
# cannot say "no Bentley in stock". Names that are also ordinary words (seat, smart, mini) are left out on purpose.
WELL_KNOWN_MAKES = ["bentley", "ferrari", "lamborghini", "rolls royce", "aston martin", "maserati", "bugatti", "mclaren", "porsche",
                    "jaguar", "volvo", "tesla", "mitsubishi", "subaru", "mazda", "suzuki", "isuzu", "peugeot", "renault", "citroen",
                    "jeep", "dodge", "cadillac", "lincoln", "genesis", "infiniti", "lexus", "acura", "volkswagen", "hyundai", "kia"]
COLORS = {"white": "white", "black": "black", "silver": "silver", "grey": "grey", "gray": "grey", "red": "red", "blue": "blue",
          "brown": "brown", "green": "green", "yellow": "yellow", "orange": "orange", "gold": "gold", "beige": "beige", "maroon": "maroon"}
KM_TOKENS = {"km", "kms", "kilometre", "kilometres", "kilometer", "kilometers", "miles"}
# "No preference" answers to a narrowing question ("what's your budget?" -> "any budget") mean: stop asking, read out the matches.
_SHOW_ME = re.compile(r"\b(?:just show|show me|read (?:them|it)|go ahead|any of them|whichever|does not matter|do not mind|"
                      r"any (?:budget|make|model|mileage|kilometres|price|one|of those)|anything|"
                      r"no (?:particular |specific )?(?:preference|budget|limit|make|mileage)|not fussy|not sure)\b|^(?:any|either)$")
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
@lru_cache(maxsize=512)
def spoken_form(model: str) -> str:
    return " ".join(tokens_of(model))


def find_vehicle(tokens: list[str], vocab: dict[str, list[str]]) -> dict:
    """{'make': ..., 'model': ...} (either may be absent) found in the tokens, with fuzzy matching on longer names."""
    out: dict = {}
    text = f" {' '.join(tokens)} "
    for alias, make in MAKE_ALIASES.items():
        if f" {alias} " in text and make in vocab:
            out["make"] = make
    grams = [" ".join(tokens[i:i + n]) for n in (1, 2, 3) for i in range(len(tokens) - n + 1)]
    makes = list(vocab) + [m for m in WELL_KNOWN_MAKES if m not in vocab]
    if "make" not in out:
        exact = [m for m in makes if f" {m} " in text]
        if exact:
            out["make"] = max(exact, key=len)
        else:
            # Fuzzy matching only for names of 5+ letters, same first letter and a similar length: "betley" is a Bentley, but "afford"
            # must not become "ford" and "that" must not become "audi". Short makes (ford, kia, bmw, audi) match exactly or not at all.
            def candidates(g: str) -> list[str]:
                return [m for m in makes if len(m) >= 5 and m[0] == g[0] and abs(len(m) - len(g)) <= 2]

            scored = [(g, process.extractOne(g, candidates(g), scorer=fuzz.ratio, score_cutoff=80) if candidates(g) else None)
                      for g in grams if len(g) >= 4]
            scored = [(g, m) for g, m in scored if m]
            if scored:
                out["make"] = max(scored, key=lambda x: x[1][1])[1][0]
    flat = [(model, make) for make, models in vocab.items() for model in models]
    # Models with digits ("m3", "320i", "c 300") are matched in the form the number normaliser gives the caller's words ("m three"), as well as as written
    hits = [(m, mk) for m, mk in flat if (f" {m} " in text or (any(ch.isdigit() for ch in m) and f" {spoken_form(m)} " in text))
            and (len(m) > 2 or "make" in out)]
    if not hits:                                              # a caller drops the trim: "the GLE" for "GLE 450", "Range Rover" for "Range Rover Sport"
        heads: dict[str, list[tuple[str, str]]] = {}
        for m, mk in flat:
            heads.setdefault(m.split()[0], []).append((m, mk))
        for head, owners in heads.items():
            if len(owners) == 1 and " " in owners[0][0] and len(head) >= 3 and f" {head} " in text and head not in vocab:
                hits.append(owners[0])
    if not hits:
        long_models = [(m, mk) for m, mk in flat if len(m) >= 6]
        for g in grams:
            if len(g) >= 6 and long_models:
                r = process.extractOne(g, [m for m, _ in long_models if m[0] == g[0]], scorer=fuzz.ratio, score_cutoff=85)
                if r:
                    hits.append((r[0], dict(long_models)[r[0]]))
                    break
    if hits:
        model, make = max(hits, key=lambda h: len(h[0]))
        if "make" not in out or out["make"] == make:
            out["make"], out["model"] = make, model
    return out


def search_filters(text: str, vocab: dict[str, list[str]]) -> dict:
    """Filters the caller asked for: make, model, body_type, color, year_min, max_price, max_km (only those actually said)."""
    toks = tokens_of(text)
    f = find_vehicle(toks, vocab)
    for t in toks:
        if t in BODY:
            f["body_type"] = BODY[t]
        elif t in COLORS:
            f["color"] = COLORS[t]
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


_UNFILTERABLE = re.compile(r"\b(?:automatic|manual|petrol|diesel|hybrid|electric|sunroof|leather|turbo|4wd|awd|seats|seater|doors?|"
                           r"diesel|cylinder|v6|v8|horsepower|camera|navigation|bluetooth|warranty|accident free)\b")


def mentions_unfilterable(text: str) -> bool:
    """Car attributes people ask for that the search cannot filter on yet (the agent should say so, not call it off-topic)."""
    return bool(_UNFILTERABLE.search(" ".join(tokens_of(text))))


_POINTS = re.compile(r"\b(?:first|1st|second|2nd|third|3rd|last|this|that) (?:one|car)\b|\bthe (?:first|second|third|last)\b")


def points_at_a_car(text: str) -> bool:
    """"tell me about the second one" / "that one": the caller refers to a car on screen, not a new search."""
    return bool(_POINTS.search(" ".join(tokens_of(text))))


_REQUEST_WORDS = {"do", "you", "have", "got", "show", "me", "i", "want", "need", "looking", "for", "any", "is", "there", "what", "how"}


def is_bare_vehicle(text: str) -> bool:
    """"nissan" / "toyota please" / "the patrol": a few words with no request wording, i.e. an answer to the conversation so far."""
    toks = tokens_of(text)
    return 0 < len(toks) <= 4 and not (set(toks) & _REQUEST_WORDS)


_OPTIONS_Q = re.compile(r"\b(?:what|which)\s+(?:(makes|brands|models|colou?rs|body types|types)|(?:kinds?|sorts?) of\s+(make|brand|model|colou?r|body type)s?)\b")
_OPTIONS_KEY = {"make": "make", "brand": "make", "model": "model", "color": "color", "colour": "color", "body type": "body_type", "type": "body_type"}


def asks_options(text: str) -> str | None:
    """"what makes are they?" / "which colours do you have?": a question about the OPTIONS in stock (plural: "what colour is it?" is not one).
    Returns make | model | color | body_type."""
    m = _OPTIONS_Q.search(" ".join(tokens_of(text)))
    if not m:
        return None
    word = (m.group(1) or m.group(2)).lower()
    word = word[:-1] if word.endswith("s") else word
    return _OPTIONS_KEY.get(word)


def wants_listing(text: str) -> bool:
    """'just show me' after a narrowing question: stop asking and read out the first matches."""
    return bool(_SHOW_ME.search(" ".join(tokens_of(text))))


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


CAR_FACTS = {"km", "color", "year", "fuel", "transmission", "spec", "service", "accident", "branch"}      # facts about a car (not price/availability)


def attributes_asked(text: str) -> list[str]:
    t = " ".join(tokens_of(text))
    return [k for k, rx in ATTRIBUTES if re.search(rx, t)]


_YES_NO_Q = re.compile(r"^(?:is|are|does|has|was|isn't|is not)\b|\b(?:is it|is this|is that|does it|does this|has it|was it)\b")
_FUEL = {"petrol", "diesel", "hybrid", "electric"}


def claims_in_question(text: str) -> dict[str, str]:
    """For a yes/no question about a car ("is it red", "is it automatic", "is it a 2021", "is it GCC"): what the caller guesses.
    Returns {} when the text is not a yes/no question. Keys: color, transmission, fuel, year, spec."""
    toks = tokens_of(text)
    joined = " ".join(toks)
    if not _YES_NO_Q.search(joined):
        return {}
    out: dict[str, str] = {}
    for t in toks:
        if t in COLORS:
            out["color"] = COLORS[t]
        elif t in ("automatic", "manual"):
            out["transmission"] = t
        elif t in _FUEL:
            out["fuel"] = t
        elif t == "gcc":
            out["spec"] = "GCC"
    for val, _, j in number_spans(toks):
        if len(val) == 4 and 1990 <= int(val) <= 2030 and (toks[j] if j < len(toks) else "") not in KM_TOKENS:
            out["year"] = val
    return out


_ONLY_ONE = re.compile(r"\b(?:only|just) (?:have |got )?(?:one|1|a single)\b|\bis that the only\b|\bthe only one\b|\bonly (?:have )?that\b")


def asks_if_only_one(text: str) -> bool:
    """"Do you only have one Camry?" / "is that the only one?"."""
    return bool(_ONLY_ONE.search(" ".join(tokens_of(text))))


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
