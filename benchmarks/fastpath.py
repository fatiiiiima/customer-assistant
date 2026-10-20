"""Deterministic NLU fast path (prototype). Rules first; the LLM only sees what rules decline.

Design rules:
  * PRECISION over coverage: return None unless confident. A wrong fast-path answer is worse
    than a 300 ms LLM call.
  * STATE-GATED: only intents that are valid in the current state can fire (same table the LLM uses).
  * Slots are extracted by code (number/date/phone parsers, make/model gazetteer with fuzzy match),
    never copied from a model, so they are grounded in the utterance by construction.
  * Makes/models come from a gazetteer. In production this is built from the inventory table
    (SELECT DISTINCT make, model), so it always matches what is actually for sale.
Prototype: moves into dealer_core/ in Phase 1 with unit tests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from nlu_cases import STATE_SPEC
from stt_eval import normalise, number_spans

# ---- gazetteer (generic popular UAE used-car list; NOT tailored to the test sets) --------------
MODELS_BY_MAKE: dict[str, list[str]] = {
    "toyota": ["land cruiser", "prado", "camry", "corolla", "yaris", "rav4", "fortuner", "hilux", "innova", "avalon", "highlander", "rush"],
    "nissan": ["patrol", "sunny", "altima", "x trail", "kicks", "pathfinder", "maxima", "navara", "juke", "armada"],
    "honda": ["accord", "civic", "city", "cr v", "pilot", "odyssey", "jazz"],
    "mitsubishi": ["pajero", "outlander", "l200", "asx", "attrage", "lancer"],
    "hyundai": ["elantra", "tucson", "santa fe", "sonata", "accent", "creta", "palisade", "i10"],
    "kia": ["sportage", "sorento", "cerato", "picanto", "seltos", "telluride", "optima"],
    "mazda": ["mazda 3", "mazda 6", "cx 5", "cx 9", "cx 30"],
    "lexus": ["rx", "es", "lx", "nx", "is", "gx", "ls"],
    "infiniti": ["qx80", "qx60", "q50", "qx50"],
    "mercedes": ["c class", "e class", "s class", "gle", "glc", "g class", "a class", "cla"],
    "bmw": ["x5", "x3", "x6", "3 series", "5 series", "7 series", "x1"],
    "audi": ["a4", "a6", "q5", "q7", "a3", "q3"],
    "porsche": ["cayenne", "macan", "panamera", "911"],
    "land rover": ["range rover", "defender", "discovery", "evoque", "velar"],
    "ford": ["mustang", "explorer", "edge", "f150", "expedition", "territory"],
    "chevrolet": ["tahoe", "suburban", "malibu", "camaro", "silverado", "captiva"],
    "jeep": ["wrangler", "grand cherokee", "compass", "gladiator"],
    "gmc": ["yukon", "sierra", "terrain"],
    "tesla": ["model 3", "model y", "model s", "model x"],
}
MAKE_ALIASES = {"mercedes benz": "mercedes", "benz": "mercedes", "merc": "mercedes", "chevy": "chevrolet",
                "rangerover": "land rover", "range rover": "land rover", "landrover": "land rover"}
BODY = {"suv": "suv", "suvs": "suv", "crossover": "suv", "4x4": "suv", "sedan": "sedan", "saloon": "sedan",
        "hatchback": "hatchback", "hatch": "hatchback", "coupe": "coupe", "pickup": "pickup",
        "pickups": "pickup", "truck": "pickup", "trucks": "pickup"}

MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
DAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
NUMW = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
DATE_RE = re.compile(rf"\b(day after tomorrow|tomorrow|today|tonight|this weekend|next week|(?:next |this )?(?:{DAYS})|"
                     rf"\d{{1,2}}(?:st|nd|rd|th)?(?: of)? (?:{MONTHS})|(?:{MONTHS}) \d{{1,2}}(?:st|nd|rd|th)?)\b")
TIME_RE = re.compile(rf"\b(\d{{1,2}}(?::\d{{2}})?\s?(?:am|pm)|(?:{NUMW}|\d{{1,2}})(?: (?:fifteen|thirty|forty five|o clock))?"
                     rf" (?:in the )?(?:morning|afternoon|evening|night)|noon|midday|morning|afternoon|evening)\b")

YES = re.compile(r"^(yes|yeah|yep|yup|sure|ok|okay|correct|right|exactly|absolutely|definitely|perfect|fine|deal|done|agreed|good|great|"
                 r"that works|that is right|that is correct|sounds good|let us do it|go ahead|works for me|please do|ok deal|alright)\b")
NO = re.compile(r"^(no|nope|nah|not really|no thanks|i do not think so|that is wrong|that is not right|that is incorrect|i do not want)\b|"
                r"\b(can not make it|cannot make it|not available|not free|does not work|will not work|do not think so)\b")
FILLER = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "thanks", "thank", "you", "that", "is", "it", "its", "right",
          "correct", "works", "for", "me", "sounds", "good", "great", "perfect", "deal", "let", "us", "do", "go", "ahead", "fine",
          "exactly", "absolutely", "definitely", "done", "agreed", "alright", "so", "then", "and", "well", "the", "a"}

RULES: list[tuple[str, re.Pattern]] = [
    ("angry", re.compile(r"\b(useless|ridiculous|rip ?off|waste (?:of )?my time|stupid|idiots?|unacceptable|terrible|awful|fed up|sick of|"
                         r"scam|liars?|nonsense|hate|damn|shut up|crap|joke)\b")),
    ("claim_authority", re.compile(r"\bi (?:am|m) (?:the |a |an )?(?:manager|owner|director|boss|ceo|staff|employee|supervisor)\b|"
                                   r"\b(?:owns?|runs?) (?:the |this )?company\b|\b(?:staff|employee|manager|dealer|trade) (?:price|rate|discount)\b|"
                                   r"\bi work (?:for|at|with) (?:you|the company|this company)\b")),
    ("request_human", re.compile(r"\b(?:speak|talk|connect|transfer|put|get)\b.{0,25}\b(?:person|human|agent|someone|somebody|manager|"
                                 r"supervisor|representative|salesperson|salesman|sales|operator)\b|\b(?:real|live) person\b|\bhuman\b|\boperator\b")),
    ("goodbye", re.compile(r"\b(?:bye|goodbye|see you|that is all|have a (?:good|nice|great)|talk later)\b")),
    ("cancel_booking", re.compile(r"\b(?:cancel|forget (?:it|about)|never ?mind|call it off|scrap)\b")),
    ("reschedule", re.compile(r"\b(?:reschedule|change (?:it|the|my)|move (?:it|the|my)|another (?:day|time)|different (?:day|time)|instead|postpone)\b")),
    ("ask_policy", None),          # handled in code (topic slot)
    ("send_details", re.compile(r"\b(?:send|mail|email|e mail|whatsapp|share|text|forward)\b.{0,30}\b(?:photos?|pictures?|details?|specs?|link|info|information|brochure)\b|"
                                r"\b(?:photos?|pictures?|details)\b.{0,20}\b(?:whatsapp|email|message)\b|\bmail me\b")),
    ("book_test_drive", re.compile(r"\b(?:test ?drive|try (?:it|the car|out)|drive it|take it for a (?:spin|drive)|see it in person|"
                                   r"come (?:and )?(?:see|view|look|check)|view(?:ing)? the car|visit)\b")),
    ("ask_lowest_price", re.compile(r"\b(?:lowest|best|final|bottom|cheapest|minimum|rock bottom|last)\b.{0,20}\b(?:price|offer|you can (?:do|give|go)|figure)\b|"
                                    r"\bhow low (?:can|could)\b|\bwhat (?:is |would )?(?:your )?(?:best|final|lowest)\b|\bcan you do better\b|\bany discount\b")),
]
TOPICS = {"warranty": r"warrant|guarantee", "financing": r"financ|loan|installment|instalment|monthly payment|bank|mortgage|credit",
          "inspection": r"inspect|mechanic|certif", "trade_in": r"trade ?in|exchange|my (?:old|current) car|sell my car",
          "documents": r"documents?|papers?|what (?:do i|should i) bring|emirates id|requirements"}
INFO_RE = re.compile(r"\b(?:how many (?:km|kilomet\w+|miles|owners|keys)|mileage|colou?r|accident\w*|damage\w*|paint\w*|service history|history|"
                     r"gcc|spec|engine|gearbox|transmission|fuel|petrol|diesel|hybrid|which year|what year|model year|owners?|sunroof|"
                     r"leather|cylinders|horsepower|registration|still available|which branch|where is it|how many kilomet\w+)\b")
OFFER_CUE = re.compile(r"\b(?:offer|pay|give you|take|do it for|do|how about|what about|go for|can you do|would you do|cash|for)\b")
SEARCH_CUE = re.compile(r"\b(?:looking for|do you have|have any|any|show me|i want|i need|i am after|after a|interested in|budget|"
                        r"something|cheap|find me|searching)\b")
PRICE_CUE = re.compile(r"\b(?:under|below|less than|up to|max|maximum|budget|around|about|within|not more than|upto)\b")
KM_TOKENS = {"km", "kms", "kilometre", "kilometres", "kilometer", "kilometers", "miles", "kilometers", "k"}


CONTRACTIONS = [(r"won't", "will not"), (r"can't", "can not"), (r"cannot", "can not"), (r"n't", " not"), (r"let's", "let us"),
                (r"'m", " am"), (r"'re", " are"), (r"'ll", " will"), (r"'ve", " have"), (r"'d", " would"), (r"'s", " is")]


def expand(text: str) -> str:
    t = text.lower().replace("’", "'")
    for pat, rep in CONTRACTIONS:
        t = re.sub(pat, rep, t)
    return t


@dataclass
class Fast:
    intent: str
    slots: dict
    rule: str


def _find_vehicle(tokens: list[str]) -> dict:
    """Fuzzy-match make/model in the utterance (tolerates STT errors like 'toyode')."""
    slots: dict = {}
    text = " ".join(tokens)
    for alias, make in MAKE_ALIASES.items():
        if f" {alias} " in f" {text} ":
            slots["make"] = make
    grams = [" ".join(tokens[i:i + n]) for n in (1, 2, 3) for i in range(len(tokens) - n + 1)]
    all_makes = list(MODELS_BY_MAKE)
    if "make" not in slots:
        scored = [(g, process.extractOne(g, all_makes, scorer=fuzz.ratio, score_cutoff=80)) for g in grams if len(g) >= 4]
        scored = [(g, m) for g, m in scored if m]
        if scored:
            slots["make"] = max(scored, key=lambda x: x[1][1])[1][0]
    # model: exact (word-boundary) match first, then fuzzy for longer names
    flat = [(model, make) for make, models in MODELS_BY_MAKE.items() for model in models]
    padded = f" {text} "
    hits = [(m, mk) for m, mk in flat if f" {m} " in padded and (len(m) > 2 or "make" in slots)]
    if not hits:
        long_models = [(m, mk) for m, mk in flat if len(m) >= 6]
        for g in grams:
            if len(g) >= 6:
                r = process.extractOne(g, [m for m, _ in long_models], scorer=fuzz.ratio, score_cutoff=88)
                if r:
                    hits.append((r[0], dict(long_models)[r[0]]))
                    break
    if hits:
        model, make = max(hits, key=lambda h: len(h[0]))
        slots.setdefault("make", make)
        if slots["make"] == make:
            slots["model"] = model
    return slots


def _search_slots(raw: str, toks: list[str], spans) -> dict:
    slots = _find_vehicle(toks)
    for t in toks:
        if t in BODY:
            slots["body_type"] = BODY[t]
    for val, i, j in spans:
        n = int(val) if val.isdigit() else 0
        nxt = toks[j] if j < len(toks) else ""
        prev = " ".join(toks[max(0, i - 3):i])
        if nxt in KM_TOKENS:
            slots["max_km"] = n
        elif 1990 <= n <= 2030 and len(val) == 4:
            slots["year_min"] = n
        elif n >= 5000 and not val.startswith("0"):
            slots["max_price"] = n
    return slots


def _contact(raw: str, spans) -> dict:
    slots: dict = {}
    for val, _, _ in spans:
        if val.isdigit() and 9 <= len(val) <= 12 and val[0] in "09":
            slots["phone"] = val
    m = re.search(r"(?:my name is|this is|i am|i m|it is|its|it s|name is)\s+([a-z]+(?: [a-z]+){0,3}?)(?=\s*(?:,|\.|and\b|my\b|phone\b|number\b|call\b|$|\d)|"
                  rf"\s(?:zero|oh|{NUMW})\b)", raw)
    if m:
        slots["name"] = m.group(1).strip()
    return slots


def classify(state: str, car: str | None, text: str) -> Fast | None:
    """Return a confident deterministic result, or None to defer to the LLM."""
    allowed = set(STATE_SPEC[state][0])
    text = expand(text)
    raw = re.sub(r"\s+", " ", re.sub(r"[^\w\s:']", " ", text.replace("'", " ").replace(",", ", "))).strip()
    raw = re.sub(r"(?<=\d) (?=\d)", " ", raw)
    toks = normalise(text).split()
    spans = number_spans(toks)
    norm = " ".join(toks)

    def ok(intent: str, slots: dict | None = None, rule: str = "") -> Fast | None:
        return Fast(intent, slots or {}, rule or intent) if intent in allowed else None

    # 1. content rules (priority order)
    for intent, pat in RULES:
        if intent == "ask_policy":
            for topic, rx in TOPICS.items():
                if re.search(rx, norm):
                    return ok("ask_policy", {"topic": topic})
            continue
        if pat and pat.search(norm):
            r = ok(intent)
            if r:
                if intent in ("reschedule",):
                    r.slots = {k: v for k, v in (("date", (DATE_RE.search(raw) or [None, None])[1]), ("time", (TIME_RE.search(raw) or [None, None])[1])) if v}
                return r
    # 2. slots by state
    if state in ("slot", "confirm"):
        d, t = DATE_RE.search(raw), TIME_RE.search(raw)
        if (d or t) and state == "slot":
            return ok("give_slot", {k: m.group(1) for k, m in (("date", d), ("time", t)) if m})
    if state == "contact":
        s = _contact(raw, spans)
        if s:
            return ok("give_contact", s)
    # 3. money
    money = [(v, i, j) for v, i, j in spans if v.isdigit() and int(v) >= 1000 and not v.startswith("0")
             and not (1990 <= int(v) <= 2030 and len(v) == 4) and (toks[j] if j < len(toks) else "") not in KM_TOKENS]
    vehicle = _find_vehicle(toks)
    searchish = bool(vehicle) or any(t in BODY for t in toks)
    if money and state == "negotiating" and not searchish:
        return ok("make_offer", {"offer_amount": int(money[0][0])})
    if money and state == "answering" and not searchish and OFFER_CUE.search(norm):
        return ok("make_offer", {"offer_amount": int(money[0][0])})
    # 4. search
    has_km = any((toks[j] if j < len(toks) else "") in KM_TOKENS for _, _, j in spans)
    if searchish or (state in ("interest", "greeting") and spans and (SEARCH_CUE.search(norm) or has_km or PRICE_CUE.search(norm))):
        s = _search_slots(raw, toks, spans)
        if s:
            return ok("search_cars", s)
    # 5. car questions
    if INFO_RE.search(norm) and (car or state == "answering"):
        return ok("ask_car_info")
    # 6. short yes / no / greeting (only when the whole utterance is basically that)
    words = norm.split()
    if YES.search(norm) and all(w in FILLER for w in words):
        return ok("confirm_yes")
    if NO.search(norm) and len(words) <= 10:
        return ok("confirm_no")
    if re.match(r"^(hi|hello|hey|good (morning|afternoon|evening)|salam|assalam\w*|marhaba|hiya)\b", norm) and len(words) <= 4:
        return ok("greet")
    return None
