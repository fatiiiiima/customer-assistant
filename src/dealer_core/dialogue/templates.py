"""Reply wording. All spoken text lives here (the LLM never writes a reply in the main flow), so tone and the
"subject to confirmation" language for prices can be reviewed in one place. Every function returns 1-2 short sentences."""
from __future__ import annotations

from datetime import datetime

from ..models import CarView
from ..tools.phones import speak_phone

OPENING = "Hello, welcome to the dealership. What kind of car are you looking for?"
GOODBYE = "Thanks for calling. Have a great day!"
HANDOFF_WAIT = "A colleague will be with you shortly."
TOPIC_LABEL = {"warranty": "warranty", "financing": "financing", "inspection": "inspection", "trade_in": "trade-in", "documents": "documents"}


def cap(text: str) -> str:
    """Capitalise the first letter only (str.capitalize() would lower-case acronyms such as GCC)."""
    return text[:1].upper() + text[1:]


def money(n: int) -> str:
    return f"{n:,} dirhams"


def car_name(c: CarView) -> str:
    return f"{c.year} {c.make} {c.model}"


def when_short(dt: datetime, now: datetime) -> str:
    """'Thursday at 4:00 PM' within a week, otherwise 'Saturday, 16 March at 4:00 PM'."""
    h = dt.hour % 12 or 12
    clock = f"{h}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"
    days = (dt.date() - now.astimezone(dt.tzinfo).date()).days
    if days == 0:
        return f"today at {clock}"
    if days == 1:
        return f"tomorrow at {clock}"
    if days < 7:
        return f"{dt:%A} at {clock}"
    return f"{dt:%A}, {dt.day} {dt:%B} at {clock}"


def when_full(dt: datetime) -> str:
    h = dt.hour % 12 or 12
    return f"{dt:%A}, {dt.day} {dt:%B} at {h}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def describe_filters(f: dict) -> str:
    """'SUVs under 150,000 dirhams' / 'Toyota Land Cruisers from 2020' / 'cars'."""
    noun = {"suv": "SUVs", "sedan": "sedans", "hatchback": "hatchbacks", "coupe": "coupes", "pickup": "pickups"}.get(f.get("body_type", ""), "cars")
    if f.get("model"):
        noun = f"{f.get('make', '').title()} {f['model'].title()}s".strip()
    elif f.get("make"):
        noun = f"{f['make'].title()} {noun}" if noun != "cars" else f"{f['make'].title()} cars"
    if f.get("color"):
        noun = f"{f['color']} {noun}"
    parts = [noun]
    if f.get("year_min"):
        parts.append(f"from {f['year_min']}")
    if f.get("max_price"):
        parts.append(f"under {money(f['max_price'])}")
    if f.get("max_km"):
        parts.append(f"with under {f['max_km']:,} kilometres")
    return " ".join(parts)


def cars_listed(cars: list[CarView]) -> str:
    items = [f"a {car_name(c)} at {money(c.price)}" for c in cars]
    if len(items) == 1:
        c = cars[0]
        return (f"I have a {car_name(c)} with {c.km:,} kilometres at {money(c.price)}. "
                "Would you like to hear more, or book a test drive?")
    joined = ", ".join(items[:-1]) + f" and {items[-1]}"
    return f"I have {joined}. Which one would you like to hear more about?"


def narrowing_situation(total: int, f: dict, asked: int = 0) -> tuple[str, dict]:
    """Which narrowing situation applies (see phrasebook.py) and the values to fill into it."""
    desc = describe_filters(f)
    if asked >= 1:
        sid = "narrow_again"
    elif not f.get("make") and not f.get("model"):
        sid = "narrow_make_mileage"
    elif not f.get("max_price"):
        sid = "narrow_budget"
    else:
        sid = "narrow_year_mileage"
    return sid, {"total": total, "desc": desc}


def narrowing_question(total: int, f: dict, asked: int = 0) -> str:
    desc = describe_filters(f)
    if asked >= 1:      # second time round: do not repeat the same words
        return (f"That's still {total} {desc}. Tell me a make, a model or a mileage limit, "
                "or say 'just show me' and I'll read out the first three.")
    if not f.get("make") and not f.get("model"):
        return f"I have {total} {desc}. Do you prefer a particular make, or a maximum mileage?"
    if not f.get("max_price"):
        return f"I have {total} {desc}. What's your budget?"
    return f"I have {total} {desc}. Is there a year or mileage you'd like to stay within?"


CANNOT_FILTER = ("I can't search on that, only on make, model, body type, colour, budget, year and mileage. "
                 "What would you like to change?")


def no_match(f: dict) -> str:
    return f"I don't have any {describe_filters(f)} right now. Would you like to change the budget or the type of car?"


def car_summary(c: CarView, branch_name: str) -> str:
    return (f"It's a {c.color} {car_name(c)} with {c.km:,} kilometres, priced at {money(c.price)}, at our {branch_name}. "
            "What else would you like to know?")


def attribute_clause(key: str, c: CarView, branch_name: str) -> str | None:
    return {
        "km": f"it has {c.km:,} kilometres", "color": f"it's {c.color}", "year": f"it's a {c.year} model",
        "price": f"it's priced at {money(c.price)}", "fuel": f"it runs on {c.fuel}",
        "transmission": f"it has {'an' if c.transmission.startswith('a') else 'a'} {c.transmission} gearbox",
        "spec": f"it's {c.regional_spec} spec",
        "service": "it has a full service history" if c.service_history else "there is no service history on file",
        "branch": f"it's at our {branch_name}",
        "availability": {"available": "it's still available", "reserved": "it's currently reserved", "sold": "it has been sold"}[c.status.value],
    }.get(key)
