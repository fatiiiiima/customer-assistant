"""Answers the code writes straight from the website's data: car details, showroom details, comparisons. No model, so every fact is exact.

Each function returns the sentences to say, or None when the question is not one it answers (the model then gets the facts instead).
"""
from __future__ import annotations

import re

from ..dialogue import templates as T
from ..models import CarView
from ..site.details import CarDetails, SiteInfo

DRIVE = {"AWD": "all-wheel drive", "4WD": "four-wheel drive", "RWD": "rear-wheel drive", "FWD": "front-wheel drive"}

# topic -> what a caller says about it
TOPICS: dict[str, re.Pattern] = {
    "warranty": re.compile(r"\bwarrant\w*|\bguarantee\w*"),
    "accident": re.compile(r"\baccident\w*|\bcrash\w*|\bdamage\w*|\bclean (?:title|history)\b|\brepainted\b|\b(?:its|the|car's|the car's) history\b"),
    "inspection": re.compile(r"\binspect\w*|\bcondition\b|\bchecked\b|\bmechanical\w*"),
    "owners": re.compile(r"\bowners?\b|\bowned\b|\bhands\b"),
    "service": re.compile(r"\bservice (?:history|record\w*)\b|\bserviced\b|\bmaintenance\b|\bmaintained\b|\b(?:its|the|car's|the car's) history\b"),
    "engine": re.compile(r"\bengine\b|\bcylinders?\b|\bv6\b|\bv8\b|\bturbo\b|\blitre engine\b"),
    "power": re.compile(r"\bhorse ?power\b|\bhp\b|\bpower\w*\b|\btorque\b|\bbhp\b"),
    "drive": re.compile(r"\b(?:all|four|rear|front)[ -]wheel\b|\bawd\b|\b4wd\b|\b4x4\b|\bfour by four\b|\bdrivetrain\b|\bwheel drive\b"),
    "speed": re.compile(r"\b0 to 100\b|\bzero to (?:a )?hundred\b|\bnought to\b|\baccelerat\w*|\btop speed\b|\bhow fast\b|\bquick\b"),
    "economy": re.compile(r"\bfuel (?:economy|consumption|efficien\w*|usage)\b|\bconsumption\b|\bmpg\b|\bper (?:100|hundred)\b|\beconomical\b|"
                          r"\bfuel efficient\b|\bthirsty\b|\bpetrol does it use\b|\buse a lot of (?:fuel|petrol)\b"),
    "boot": re.compile(r"\bboot\b|\btrunk\b|\bluggage\b|\bcargo\b|\blitres of space\b|\bstorage\b"),
    "seats": re.compile(r"\bhow many (?:seats|people|passengers)\b|\bseater\b|\bseats? (?:seven|eight|five|7|8|5)\b|\bthird row\b"),
    "doors": re.compile(r"\bdoors?\b"),
    "interior": re.compile(r"\binterior\b|\bupholster\w*|\bleather\b|\bfabric\b|\bcabin\b|\binside\b"),
    "spec": re.compile(r"\bgcc\b|\bspecs?\b|\bimport\w*\b|\bamerican spec\b|\bus spec\b|\bregional\b"),
    "gearbox": re.compile(r"\bgear ?box\b|\btransmission\b|\bautomatic\b|\bmanual\b|\bgears\b"),
    "fuel": re.compile(r"\b(?:petrol|diesel|hybrid|electric|ev|plug in)\b|\bwhat fuel\b|\bfuel type\b"),
    "range": re.compile(r"\b(?:range|how far can it (?:go|drive)|on a (?:full )?charge|battery)\b"),
    "features": re.compile(r"\bfeatures?\b|\bextras?\b|\bequipment\b|\boptions\b|\bwhat does it come with\b|\bwhat comes with\b|\bloaded\b"),
    "about": re.compile(r"\btell me (?:more|about it|everything)\b|\bdescribe\b|\bwhat is it like\b|\bwhat's it like\b|\bany good\b(?! for)|\bworth (?:it|buying)\b|"
                       r"\b(?:know|hear|learn) more\b|\bmore (?:about (?:it|the car|this|that)|details|info\w*)\b|"
                       r"\b(?:anything|what) else\b.*\b(?:know|about (?:it|the car|this (?:one|car)|that (?:one|car)))\b|"
                       r"\bwhat else (?:can|could|should|do) (?:you|i)\b.*\b(?:tell|say|share|know)\b|\b(?:tell|give) me something else\b"),
}

# "does it have a sunroof?": what a caller says -> words to look for in the feature list
FEATURE_WORDS: dict[str, tuple[str, ...]] = {
    "sunroof": ("sunroof", "panoramic", "roof"), "panoramic roof": ("panoramic", "sunroof"), "carplay": ("carplay",), "apple carplay": ("carplay",),
    "android auto": ("android",), "camera": ("camera", "view monitor", "around-view"), "reverse camera": ("camera", "reverse"),
    "360 camera": ("360", "around-view", "surround"), "parking sensors": ("parking", "sensor"), "heated seats": ("heated",),
    "ventilated seats": ("ventilated", "cooled"), "cooled seats": ("ventilated", "cooled"), "cruise control": ("cruise",),
    "adaptive cruise": ("adaptive cruise", "cruise"), "blind spot": ("blind-spot", "blind spot"), "lane assist": ("lane",),
    "navigation": ("navigation", "nav"), "head up display": ("head-up", "head up"), "wireless charging": ("wireless",),
    "keyless": ("keyless", "smart entry", "comfort access"), "premium sound": ("bose", "harman", "burmester", "bang", "meridian", "audio", "sound"),
    "bose": ("bose",), "tow": ("tow",), "third row": ("seven seats", "eight seats", "third row", "7 seats", "8 seats"),
    "massage seats": ("massage",), "digital cluster": ("digital",), "ambient lighting": ("ambient",), "autopilot": ("autopilot",),
}
_ASKS_HAVE = re.compile(r"\b(?:does it have|has it got|does it come with|is there an?|do you get|with an?|it got)\b")


def spoken_consumption(c: str) -> str:
    m = re.match(r"\s*([\d.]+)\s*L/100\s*km", c)
    if m:
        return f"{m.group(1)} litres per hundred kilometres"
    m = re.match(r"\s*([\d.]+)\s*kWh/100\s*km", c)
    return f"{m.group(1)} kilowatt hours per hundred kilometres" if m else c


_HAS_VERB = re.compile(r"\b(?:is|are|was|were|has|have|had|comes?|offers?|gives?|makes?|delivers?|brings?|feels?|drives?|handles?|"
                       r"rides?|sits|built|designed|packed|loaded)\b|'s\b", re.I)


def blurb_sentence(blurb: str) -> str:
    """The website's one-line description as something a person says: "A thunderous V8 super-SUV ..." -> "It's a thunderous V8 super-SUV ...";
    "Sharp styling and a punchy turbo engine ..." -> "It offers sharp styling and a punchy turbo engine ..."; a full sentence stays as it is."""
    b = blurb.strip()
    if not b:
        return b
    b = b if b.endswith((".", "!")) else b + "."
    if _HAS_VERB.search(b):
        return b
    first = b.split(" ", 1)[0]
    keep = sum(ch.isupper() for ch in first) >= 2 or any(ch.isdigit() for ch in first)       # "SUV ...", "V8 ...", "CX-5 ..." keep their capitals
    low = b if keep else b[:1].lower() + b[1:]
    return ("It's " if first.lower() in ("a", "an") else "It offers ") + low


def topics_in(words: str) -> list[str]:
    """The topics asked about. "Tell me more about the engine" is about the engine: "tell me more" only carries the question, so the generic "about" (the car's
    blurb, the next group not told yet) counts only when nothing more specific was asked. (28 Oct: it answered the engine and then added the blurb.)"""
    found = [k for k, p in TOPICS.items() if p.search(words)]
    if "about" in found and len(found) > 1:
        found.remove("about")
    return found


def asked_feature(words: str) -> str | None:
    if not _ASKS_HAVE.search(words):
        return None
    return next((k for k in sorted(FEATURE_WORDS, key=len, reverse=True) if re.search(rf"\b{re.escape(k)}s?\b", words)), None)


def _a(noun: str) -> str:
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun


_COUNTABLE = {"camera", "sunroof", "roof", "display", "monitor", "system", "cluster", "pack", "box", "bodykit", "kit", "gearbox", "screen", "spoiler",
               "exhaust", "hitch", "differential", "tailgate", "fridge", "cockpit"}


def feature_phrase(f: str) -> str:
    """"Reverse camera" -> "a reverse camera"; a brand name stays as it is ("Toyota Safety Sense", "Apple CarPlay"); plurals take no article."""
    words = f.split()
    proper = sum(w[:1].isupper() for w in words) > 1 or any(ch.isupper() for w in words for ch in w[1:])
    if proper:
        return f
    low = f[0].lower() + f[1:]
    countable = words[-1].lower() in _COUNTABLE
    return _a(low) if countable and not low[:1].isdigit() else low


def spoken_engine(e: str) -> str:
    return re.sub(r"(\d+(?:\.\d+)?)L\b", r"\1-litre", e)


def _feature_list(d: CarDetails, n: int = 3) -> str:
    f = [feature_phrase(x) for x in d.features[:n]]
    return f[0] if len(f) == 1 else ", ".join(f[:-1]) + " and " + f[-1] if f else ""


# ------------------------------------------------------------------------------------------------ spec-sheet values as spoken
# The database holds what the dealership typed ("Accident-free", "12 months / 20,000 km"); these say it.
def spoken(x: str) -> str:
    """"Accident-free" -> "accident-free"; "GCC specs" stays."""
    return x[:1].lower() + x[1:] if len(x) > 1 and x[1:2].islower() else x


def spoken_warranty(w: str) -> str:
    """"12 months / 20,000 km" -> "12 months or 20,000 kilometres"."""
    return re.sub(r"\bkms?\b", "kilometres", re.sub(r"\s*/\s*", " or ", w.strip()))


def _service(x: str) -> str:
    """"Full dealer service history" -> "a full dealer service history"; "No service history" -> "no service history"."""
    s = spoken(x)
    return s if s.lower().startswith("no ") else _a(s)


def _accident(x: str) -> str:
    s = spoken(x)
    return f"It's {s}" if re.match(r"(?:accident[- ]free|no accident)", s, re.I) else f"Its accident record says: {s}"


def _known(*values) -> bool:
    return all(v is not None and v != "" and v != () for v in values)


# What to say when the spec sheet does not have it (owner, 30 Oct: an unknown fact is said to be unknown, never guessed)
NOT_ON_RECORD = {"warranty": "the warranty", "accident": "its accident history", "inspection": "its inspection result", "owners": "how many owners it's had",
                 "service": "its service history", "engine": "its engine details", "power": "its power figures", "drive": "its drivetrain",
                 "speed": "its performance figures", "economy": "its fuel consumption", "boot": "its boot size", "seats": "how many seats it has",
                 "doors": "how many doors it has", "interior": "its interior", "spec": "its regional spec", "gearbox": "its gearbox", "fuel": "its fuel type",
                 "features": "its feature list", "about": "a description of it", "range": "its range"}


def not_on_record(topic: str) -> str:
    return f"I don't have {NOT_ON_RECORD.get(topic, 'that')} on record for this car."


def _topic_sentence(t: str, car: CarView, d: CarDetails) -> str | None:
    """The sentence answering topic `t` from the spec sheet; None when the sheet does not record it."""
    if t == "warranty":
        return f"It comes with an AGMC warranty of {spoken_warranty(d.warranty)}." if _known(d.warranty) else None
    if t == "accident":
        if not _known(d.accident):
            return None
        return _accident(d.accident) + (f", and it passed {d.inspection} of our 150 inspection checks." if _known(d.inspection) else ".")
    if t == "inspection":
        if not _known(d.inspection):
            return None
        return f"It passed {d.inspection} of our 150 inspection checks" + (f", and {_accident(d.accident)[0].lower() + _accident(d.accident)[1:]}." if _known(d.accident) else ".")
    if t == "owners":
        if not _known(d.owners):
            return None
        return f"It's had {d.owners} previous owner{'s' if d.owners != 1 else ''}" + (f", with {_service(d.service)}." if _known(d.service) else ".")
    if t == "service":
        return f"It has {_service(d.service)}." if _known(d.service) else None
    if t == "engine":
        if not _known(d.engine):
            return None
        return f"It has {_a(spoken_engine(d.engine))} engine" + (f" with {d.power_hp} horsepower." if _known(d.power_hp) else ".")
    if t == "power":
        if not _known(d.power_hp):
            return None
        return f"It makes {d.power_hp} horsepower" + (f" and {d.torque_nm} newton metres of torque." if _known(d.torque_nm) else ".")
    if t == "drive":
        return f"It's {DRIVE.get(d.drive.upper(), d.drive)}." if _known(d.drive) else None
    if t == "speed":
        if _known(d.accel_0_100, d.top_speed):
            return f"It does nought to a hundred in {d.accel_0_100:g} seconds, with a top speed of {d.top_speed} kilometres an hour."
        if _known(d.accel_0_100):
            return f"It does nought to a hundred in {d.accel_0_100:g} seconds."
        return f"Its top speed is {d.top_speed} kilometres an hour." if _known(d.top_speed) else None
    if t == "economy":
        return f"It uses about {spoken_consumption(d.consumption)}." if _known(d.consumption) else None
    if t == "boot":
        return f"The boot holds {d.boot_litres} litres." if _known(d.boot_litres) else None
    if t == "seats":
        return f"It has {d.seats} seats." if _known(d.seats) else None
    if t == "doors":
        return f"It has {d.doors} doors." if _known(d.doors) else None
    if t == "interior":
        return f"The interior is {d.interior.lower()}." if _known(d.interior) else None
    if t == "spec":
        return f"It's a {d.spec.replace(' specs', '-spec').replace(' spec', '-spec')} car." if _known(d.spec) else None
    if t == "gearbox":
        return f"It's {spoken(d.transmission)}." if _known(d.transmission) else None
    if t == "fuel":
        return f"It's {'a hybrid' if car.fuel == 'hybrid' else 'electric' if car.fuel == 'electric' else car.fuel}." if car.fuel else None
    if t == "features":
        return f"Highlights include {_feature_list(d, 4)}." if d.features else None
    if t == "about":
        return blurb_sentence(d.blurb) if _known(d.blurb) else None
    if t == "range":
        if not _known(d.consumption):
            return None
        return ("I don't have its range figure on file, but it uses about " + spoken_consumption(d.consumption) + "."
                if car.fuel == "electric" else f"It uses about {spoken_consumption(d.consumption)}.")
    return None


def detail_answer(car: CarView, d: CarDetails | None, words: str) -> list[str] | None:
    """The answer to a question about one car's details, from its spec sheet; what the sheet does not record is said to be not on record."""
    if d is None:
        return None
    feat = asked_feature(words)
    if feat:
        hits = [f for f in d.features if any(w in f.lower() for w in FEATURE_WORDS[feat])]
        if feat == "third row" and _known(d.seats) and d.seats >= 7:
            return [f"Yes, it has {d.seats} seats."]
        if hits:
            return [f"Yes, it has {feature_phrase(hits[0])}."]
        asked = feat if feat.endswith("s") else _a(feat)
        if not d.features:
            return [f"I don't have its feature list on record, so I can't say whether it has {asked}."]
        return [f"I don't see {asked} in its listed features. It has {_feature_list(d)}."]
    out: list[str] = []
    topics = topics_in(words)
    for t in topics:
        s = _topic_sentence(t, car, d) or not_on_record(t)
        if s not in out:
            out.append(s)
    if "accident" in topics and "inspection" in topics:
        out = [x for x in out if not x.startswith("It passed")]
    return out[:2] or None


# ------------------------------------------------------------------------------------------------ not telling the same thing twice
# "tell me more", "anything else about it?", "something else about the Macan": the next group of facts that has not been told yet (27 Oct: the same blurb four times)
FACT_GROUPS = ("about", "condition", "performance", "features", "practical", "economy", "ownership", "interior")
TOPIC_GROUP = {"about": "about", "accident": "condition", "inspection": "condition", "service": "condition", "owners": "ownership", "warranty": "ownership", "spec": "ownership",
               "engine": "performance", "power": "performance", "speed": "performance", "drive": "performance", "gearbox": "performance", "economy": "economy", "range": "economy",
               "fuel": "economy", "boot": "practical", "seats": "practical", "doors": "practical", "features": "features", "interior": "interior"}


def _join(*parts: str | None) -> str | None:
    """Clauses that are known, as one sentence ("It's accident-free with a full service history"); None when none is."""
    have = [p for p in parts if p]
    if not have:
        return None
    body = ", and ".join(have)
    return body[0].upper() + body[1:] + "."


def group_sentence(group: str, d: CarDetails) -> str | None:
    """One sentence about a group of facts; None when the spec sheet records none of them."""
    if group == "about":
        return blurb_sentence(d.blurb) if _known(d.blurb) else None
    if group == "condition":
        acc = _accident(d.accident) if _known(d.accident) else None
        svc = f"{'with' if acc else 'it has'} {_service(d.service)}" if _known(d.service) else None
        head = " ".join(x for x in (acc, svc) if x) or None
        checks = f"{'it passed' if head else 'It passed'} {d.inspection} of our 150 checks" if _known(d.inspection) else None
        return _join(head, checks)
    if group == "performance":
        return _topic_sentence("engine", None, d) or _topic_sentence("power", None, d)
    if group == "features":
        return f"Highlights include {_feature_list(d, 4)}." if d.features else None
    if group == "practical":
        seats = f"It seats {d.seats}" if _known(d.seats) else None
        boot = (f"with a {d.boot_litres}-litre boot" if seats else f"It has a {d.boot_litres}-litre boot") if _known(d.boot_litres) else None
        return ", ".join(x for x in (seats, boot) if x) + "." if (seats or boot) else None
    if group == "economy":
        return _topic_sentence("economy", None, d)
    if group == "ownership":
        owners = f"It's had {d.owners} previous owner{'s' if d.owners != 1 else ''}" if _known(d.owners) else None
        warranty = f"{'comes' if owners else 'It comes'} with an AGMC warranty of {spoken_warranty(d.warranty)}" if _known(d.warranty) else None
        return _join(owners, warranty)
    if group == "interior":
        return _topic_sentence("interior", None, d)
    return None


def mark_said(said: list[str], topics: list[str]) -> None:
    """Note that these topics (as asked about, see TOPICS) have been told."""
    for t in topics:
        g = TOPIC_GROUP.get(t)
        if g and g not in said:
            said.append(g)


def next_facts(d: CarDetails | None, said: list[str], prefer: tuple[str, ...] = (), limit: int = 2) -> list[str]:
    """Up to `limit` sentences from the groups of facts not yet told (those in `prefer` first), marked as told. Empty when everything has been said."""
    if d is None:
        return []
    out: list[str] = []
    for g in [*prefer, *FACT_GROUPS]:
        if g in said or g not in FACT_GROUPS or len(out) >= limit:
            continue
        line = group_sentence(g, d)
        if line:                                                        # a group the spec sheet records nothing of is not offered
            out.append(line)
            said.append(g)
    return out


# ------------------------------------------------------------------------------------------------ showroom
SHOWROOM = {
    "hours": re.compile(r"\b(?:open|opening|close|closing|hours|timings?|what time do you|until when|when are you|working days?|fridays?|weekends?)\b"),
    "where": re.compile(r"\b(?:where are you|where is (?:the|your) (?:showroom|branch|dealership)|location|located|address|directions|how do i get|"
                        r"branch(?:es)?|showrooms?|sharjah|motor city|abu dhabi|ajman)\b"),
    "contact": re.compile(r"\b(?:your (?:phone|number|whatsapp|email)|phone number for|contact|whatsapp|e ?mail|call you|reach you|hotline)\b"),
}


def showroom_topics(words: str) -> list[str]:
    return [k for k, p in SHOWROOM.items() if p.search(words)]


_DIGIT = dict(zip("0123456789", ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")))


def spoken_phone(n: str) -> str:
    """"800AGMC" -> "eight hundred, A G M C"; "+971 4 339 1555" -> "plus nine seven one, four, three three nine, one five five five"."""
    n = n.strip()
    if re.fullmatch(r"800\s*[A-Z]+", n):
        return f"eight hundred, {' '.join(n[3:].strip())}"
    groups = re.findall(r"\d+", n)
    if not groups:
        return n
    said = ", ".join(" ".join(_DIGIT[c] for c in g) for g in groups)
    return ("plus " if n.startswith("+") else "") + said


def showroom_answer(site: SiteInfo, words: str) -> list[str] | None:
    topics = showroom_topics(words)
    if not topics:
        return None
    main = site.locations[0] if site.locations else None
    others = site.locations[1:]
    hq = (main.name.split("—")[-1].split("-")[-1].strip() if main else "") or site.location
    out: list[str] = []
    if "hours" in topics:
        out.append(f"Our showroom{' on ' + hq if hq else ''} is open daily from {T.when_clock(site.open_hour)} to {T.when_clock(site.close_hour)}.")
    if "where" in topics:
        asked = next((loc for loc in others if any(w in words for w in re.findall(r"[a-z]+", loc.name.lower()) if len(w) > 4 and w != "dubai")), None)
        if asked:
            out.append(f"Yes, we have a branch in {asked.address.split(',')[0]}.")
        elif main:
            more = " and ".join(loc.address.split(",")[0] for loc in others)
            out.append(f"Our main showroom is on {hq}, in {main.address}." + (f" We also have branches in {more}." if more else ""))
    if "contact" in topics:
        if site.phone:
            out.append(f"You can call us on {spoken_phone(site.phone)}" + (f", or WhatsApp us on {spoken_phone(site.whatsapp)}" if site.whatsapp else "") + ".")
        elif site.whatsapp:
            out.append(f"You can WhatsApp us on {spoken_phone(site.whatsapp)}.")
        else:
            out.append("I don't have our phone number on record here, I'm afraid.")
    return out or None


# ------------------------------------------------------------------------------------------------ "which one is the ...?"
SUPERLATIVES = {  # caller's words -> (what is compared, larger wins?, how it is said)
    "fastest": (re.compile(r"\b(?:fastest|quickest|fast(?:er)? one|quicker)\b"), "accel", False),
    "most powerful": (re.compile(r"\b(?:most powerful|most power|strongest|most horsepower)\b"), "power", True),
    "cheapest": (re.compile(r"\b(?:cheapest|least expensive|lowest price|most affordable|cheaper one)\b"), "price", False),
    "most expensive": (re.compile(r"\b(?:most expensive|priciest|dearest)\b"), "price", True),
    "newest": (re.compile(r"\b(?:newest|latest|most recent|youngest)\b"), "year", True),
    "lowest mileage": (re.compile(r"\b(?:lowest (?:mileage|kilomet\w*)|least (?:mileage|kilomet\w*|driven)|fewest kilomet\w*)\b"), "km", False),
    "most economical": (re.compile(r"\b(?:most (?:economical|efficient|fuel efficient)|best (?:fuel|economy|mileage)|uses the least)\b"), "economy", False),
    "biggest boot": (re.compile(r"\b(?:biggest|largest|most) (?:boot|trunk|luggage|space)\b"), "boot", True),
    "best for a family": (re.compile(r"\b(?:better|best|good) (?:for (?:a |the )?famil\w+|family car)\b|\bfor (?:a |the )?family\b"), "family", True),
    "most seats": (re.compile(r"\b(?:most seats|biggest|largest|roomiest|most space)\b"), "seats", True),
}


def superlative_topic(words: str) -> str | None:
    return next((k for k, (p, _, _) in SUPERLATIVES.items() if p.search(words)), None)


def best_of(cars: list[CarView], details: dict[str, CarDetails], words: str) -> tuple[CarView, str] | None:
    """(the car, the sentence) for "which one is the fastest / cheapest / newest...?" among the cars being discussed."""
    topic = superlative_topic(words)
    if topic is None or not cars:
        return None
    _, key, larger = SUPERLATIVES[topic]

    def value(c: CarView):
        d = details.get(c.stock_id)
        if key == "price":
            return c.price
        if key == "year":
            return c.year
        if key == "km":
            return c.km
        if d is None:
            return None
        if key == "economy":
            m = re.match(r"[\d.]+", d.consumption or "")
            return float(m.group()) if m and "L/100" in d.consumption else None
        if key == "family":
            return d.seats * 1000 + d.boot_litres if _known(d.seats, d.boot_litres) else None     # seats first, then the boot
        return {"accel": d.accel_0_100, "power": d.power_hp, "boot": d.boot_litres, "seats": d.seats}[key]      # None: not recorded, left out

    scored = [(value(c), c) for c in cars if value(c) is not None]
    if not scored:
        return None
    v, best = (max if larger else min)(scored, key=lambda x: x[0])
    d = details.get(best.stock_id)
    what = {"accel": f"nought to a hundred in {v:g} seconds", "power": f"{v} horsepower", "price": T.money(v), "year": f"a {v}",
            "km": f"{v:,} kilometres", "economy": f"about {spoken_consumption(d.consumption)}" if d and d.consumption else "", "boot": f"{v} litres of boot space",
            "seats": f"{v} seats", "family": f"{d.seats} seats and a {d.boot_litres}-litre boot" if d and _known(d.seats, d.boot_litres) else ""}[key]
    among = "of these" if len(scored) > 1 else ""
    return best, f"The {topic} {among} is the {T.short_name(best)}, with {what}.".replace("  ", " ").replace(" with a ", ", a ")


# ------------------------------------------------------------------------------------------------ comparison
def compare(a: CarView, da: CarDetails | None, b: CarView, db: CarDetails | None) -> list[str]:
    """Two cars side by side in at most three sentences: the basics, then the two clearest differences."""
    na, nb = T.short_name(a), T.short_name(b)
    out = [f"The {na} is {T.money(a.price)} with {a.km:,} kilometres, and the {nb} is {T.money(b.price)} with {b.km:,}."]
    diffs: list[tuple[float, str]] = []
    if a.price != b.price:
        cheap, dear = (a, b) if a.price < b.price else (b, a)
        diffs.append((abs(a.price - b.price) / max(a.price, b.price), f"The {T.short_name(cheap)} is {T.money(dear.price - cheap.price)} cheaper."))
    if a.year != b.year:
        newer = a if a.year > b.year else b
        diffs.append((abs(a.year - b.year) / 5, f"The {T.short_name(newer)} is newer, a {newer.year}."))
    if da and db:                                                       # each difference only when both sheets record it
        if _known(da.drive, db.drive) and da.drive != db.drive and ("AWD" in (da.drive, db.drive) or "4WD" in (da.drive, db.drive)):
            x, d = (a, da) if da.drive in ("AWD", "4WD") else (b, db)
            diffs.append((0.5, f"The {T.short_name(x)} has {DRIVE.get(d.drive, d.drive)}."))
        if _known(da.seats, db.seats) and da.seats != db.seats:
            x, d = (a, da) if da.seats > db.seats else (b, db)
            diffs.append((0.6, f"The {T.short_name(x)} seats {d.seats}."))
        if _known(da.power_hp, db.power_hp) and abs(da.power_hp - db.power_hp) >= 40:
            x, d = (a, da) if da.power_hp > db.power_hp else (b, db)
            diffs.append((abs(da.power_hp - db.power_hp) / max(da.power_hp, db.power_hp), f"The {T.short_name(x)} is more powerful, with {d.power_hp} horsepower."))
        ca, cb = (re.match(r"[\d.]+", da.consumption or ""), re.match(r"[\d.]+", db.consumption or ""))
        if ca and cb and "L/100" in da.consumption and "L/100" in db.consumption and abs(float(ca.group()) - float(cb.group())) >= 1.5:
            x, d = (a, da) if float(ca.group()) < float(cb.group()) else (b, db)
            diffs.append((0.45, f"The {T.short_name(x)} uses less fuel, about {spoken_consumption(d.consumption)}."))
        if _known(da.boot_litres, db.boot_litres) and abs(da.boot_litres - db.boot_litres) >= 80:
            x, d = (a, da) if da.boot_litres > db.boot_litres else (b, db)
            diffs.append((0.3, f"The {T.short_name(x)} has the bigger boot, {d.boot_litres} litres."))
    out += [s for _, s in sorted(diffs, key=lambda x: -x[0])[:2]]
    return out
