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
    "accident": re.compile(r"\baccident\w*|\bcrash\w*|\bdamage\w*|\bclean (?:title|history)\b|\brepainted\b"),
    "inspection": re.compile(r"\binspect\w*|\bcondition\b|\bchecked\b|\bmechanical\w*"),
    "owners": re.compile(r"\bowners?\b|\bowned\b|\bhands\b"),
    "service": re.compile(r"\bservice (?:history|record\w*)\b|\bserviced\b|\bmaintenance\b|\bmaintained\b"),
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
    "features": re.compile(r"\bfeatures?\b|\bextras?\b|\bequipment\b|\boptions\b|\bwhat does it come with\b|\bwhat comes with\b|\bloaded\b"),
    "about": re.compile(r"\btell me (?:more|about it|everything)\b|\bdescribe\b|\bwhat is it like\b|\bwhat's it like\b|\bany good\b(?! for)|\bworth (?:it|buying)\b"),
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


def topics_in(words: str) -> list[str]:
    return [k for k, p in TOPICS.items() if p.search(words)]


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


def detail_answer(car: CarView, d: CarDetails | None, words: str) -> list[str] | None:
    """The answer to a question about one car's details, from its spec sheet."""
    if d is None:
        return None
    feat = asked_feature(words)
    if feat:
        hits = [f for f in d.features if any(w in f.lower() for w in FEATURE_WORDS[feat])]
        if feat == "third row" and d.seats >= 7:
            return [f"Yes, it has {d.seats} seats."]
        if hits:
            return [f"Yes, it has {feature_phrase(hits[0])}."]
        asked = feat if feat.endswith("s") else _a(feat)
        return [f"I don't see {asked} in its listed features. It has {_feature_list(d)}."]
    out: list[str] = []
    for t in topics_in(words):
        s = {
            "warranty": f"It comes with an AGMC warranty of {d.warranty}.",
            "accident": f"It's {d.accident}, and it passed {d.inspection} of our 150 inspection checks.",
            "inspection": f"It passed {d.inspection} of our 150 inspection checks, and it's {d.accident}.",
            "owners": f"It's had {d.owners} previous owner{'s' if d.owners != 1 else ''}, with a {d.service}.",
            "service": f"It has a {d.service}.",
            "engine": f"It has {_a(spoken_engine(d.engine))} engine with {d.power_hp} horsepower.",
            "power": f"It makes {d.power_hp} horsepower and {d.torque_nm} newton metres of torque.",
            "drive": f"It's {DRIVE.get(d.drive.upper(), d.drive)}.",
            "speed": f"It does nought to a hundred in {d.accel_0_100:g} seconds, with a top speed of {d.top_speed} kilometres an hour.",
            "economy": f"It uses about {spoken_consumption(d.consumption)}.",
            "boot": f"The boot holds {d.boot_litres} litres.",
            "seats": f"It has {d.seats} seats.",
            "doors": f"It has {d.doors} doors.",
            "interior": f"The interior is {d.interior.lower()}.",
            "spec": f"It's a {d.spec.replace(' specs', '-spec').replace(' spec', '-spec')} car.",
            "gearbox": f"It's {d.transmission}.",
            "fuel": f"It's {'a hybrid' if car.fuel == 'hybrid' else 'electric' if car.fuel == 'electric' else car.fuel}.",
            "features": f"Highlights include {_feature_list(d, 4)}.",
            "about": d.blurb if d.blurb.endswith(".") else d.blurb + ".",
        }[t]
        if t == "inspection" and "accident" in out:
            continue
        if s not in out:
            out.append(s)
    if "accident" in topics_in(words) and "inspection" in topics_in(words):
        out = [x for x in out if not x.startswith("It passed")]
    return out[:2] or None


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
        out.append(f"Our showroom on {hq} is open daily from {T.when_clock(site.open_hour)} to {T.when_clock(site.close_hour)}.")
    if "where" in topics:
        asked = next((loc for loc in others if any(w in words for w in re.findall(r"[a-z]+", loc.name.lower()) if len(w) > 4 and w != "dubai")), None)
        if asked:
            out.append(f"Yes, we have a branch in {asked.address.split(',')[0]}.")
        elif main:
            more = " and ".join(loc.address.split(",")[0] for loc in others)
            out.append(f"Our main showroom is on {hq}, in {main.address}." + (f" We also have branches in {more}." if more else ""))
    if "contact" in topics:
        out.append(f"You can call us on {spoken_phone(site.phone)}" + (f", or WhatsApp us on {spoken_phone(site.whatsapp)}" if site.whatsapp else "") + ".")
    return out or None


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
    if da and db:
        if da.drive != db.drive and ("AWD" in (da.drive, db.drive) or "4WD" in (da.drive, db.drive)):
            x, d = (a, da) if da.drive in ("AWD", "4WD") else (b, db)
            diffs.append((0.5, f"The {T.short_name(x)} has {DRIVE.get(d.drive, d.drive)}."))
        if da.seats != db.seats:
            x, d = (a, da) if da.seats > db.seats else (b, db)
            diffs.append((0.6, f"The {T.short_name(x)} seats {d.seats}."))
        if abs(da.power_hp - db.power_hp) >= 40:
            x, d = (a, da) if da.power_hp > db.power_hp else (b, db)
            diffs.append((abs(da.power_hp - db.power_hp) / max(da.power_hp, db.power_hp), f"The {T.short_name(x)} is more powerful, with {d.power_hp} horsepower."))
        ca, cb = (re.match(r"[\d.]+", da.consumption), re.match(r"[\d.]+", db.consumption))
        if ca and cb and "L/100" in da.consumption and "L/100" in db.consumption and abs(float(ca.group()) - float(cb.group())) >= 1.5:
            x, d = (a, da) if float(ca.group()) < float(cb.group()) else (b, db)
            diffs.append((0.45, f"The {T.short_name(x)} uses less fuel, about {spoken_consumption(d.consumption)}."))
        if abs(da.boot_litres - db.boot_litres) >= 80:
            x, d = (a, da) if da.boot_litres > db.boot_litres else (b, db)
            diffs.append((0.3, f"The {T.short_name(x)} has the bigger boot, {d.boot_litres} litres."))
    out += [s for _, s in sorted(diffs, key=lambda x: -x[0])[:2]]
    return out
