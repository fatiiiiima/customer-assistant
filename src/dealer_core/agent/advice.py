"""Suggestions for a customer who describes a need instead of naming a car (5 Nov, the owner: the model should ask what they are after, or suggest cars for different
requirements, as conversational as it can be within limits).

The model decides THAT this is a need and which one (`advise family`); the cars come from the stock and the site's spec sheets, chosen here by figures, so what the model then
says is grounded: seats, boot, horsepower, 0 to 100, fuel use, kilometres, year, price against the market reference. Nothing here says a car is popular or a best seller (the
stock has no sales figures) or "great" or "reliable" (nothing records it); the facts handed to the model forbid those words too.
"""
from __future__ import annotations

import re

from ..dialogue import templates as T
from ..models import CarView
from ..site.details import CarDetails

NEEDS = ("family", "sporty", "economy", "luxury", "budget", "trip", "any")

_WORDS = (("family", r"famil|kids?|children|seven seat|7 seat|seats|spacious|space|people"), ("sporty", r"sport|fast|quick|powerful|power|performance|speed|exciting|fun to drive"),
          ("economy", r"econom|fuel|petrol cost|efficien|hybrid|electric|cheap to run|running cost|mileage per"), ("luxury", r"luxur|premium|comfort|executive|prestig|high end|nice car|impress"),
          ("budget", r"cheap|budget|afford|low price|inexpensive|first car|student|small"), ("trip", r"trip|travel|long drive|long distance|road ?trip|highway|desert|off ?road"))


def need_from_words(words: str) -> str:
    """The need in the caller's words when the model gave none (or something unknown): the first of the listed kinds that matches, else `any`."""
    for need, pattern in _WORDS:
        if re.search(pattern, words.lower()):
            return need
    return "any"


def _consumption(d: CarDetails | None) -> float | None:
    m = re.match(r"[\d.]+", (d.consumption or "") if d else "")
    return float(m.group()) if m and d and "L/100" in d.consumption else None


def _seats(d: CarDetails | None) -> int | None:
    return d.seats if d and d.seats is not None else None


def _score(need: str, c: CarView, d: CarDetails | None) -> float | None:
    """Higher is a better fit; None: the figures needed are not recorded, so the car is left out of this need."""
    if need in ("family", "trip"):
        seats = _seats(d)
        if seats is None or seats < 5:
            return None
        boot = d.boot_litres or 0
        return seats * 1000 + boot - (c.km / 100 if need == "trip" else 0) + (c.year - 2000) * (30 if need == "trip" else 0)
    if need == "sporty":
        if d and d.power_hp is not None:
            return d.power_hp - (d.accel_0_100 or 10) * 5
        return 150 if c.body_type == "coupe" else None
    if need == "economy":
        if c.fuel in ("electric", "hybrid"):
            return 1000 - c.price / 1000
        cons = _consumption(d)
        return 500 - cons * 20 if cons is not None else None
    if need == "luxury":
        return c.price + (c.year - 2015) * 1000 if c.year >= 2019 and c.body_type in ("sedan", "suv", "coupe") else None
    if need == "budget":
        return -c.price
    return None


def shortlist(cars: list[CarView], details: dict[str, CarDetails], need: str, n: int = 3) -> list[CarView]:
    """The `n` best fits for the need, one per make and model. `any` gives a spread: the cheapest, the best for a family and the most powerful."""
    if need == "any":
        out: list[CarView] = []
        for part in ("budget", "family", "sporty"):
            for c in shortlist(cars, details, part, 1):
                if all((c.make, c.model) != (o.make, o.model) for o in out):
                    out.append(c)
        return out[:n]
    scored = [(s, c) for c in cars if (s := _score(need, c, details.get(c.stock_id))) is not None]
    out, seen = [], set()
    for _, c in sorted(scored, key=lambda x: -x[0]):
        if (c.make, c.model) not in seen:
            seen.add((c.make, c.model))
            out.append(c)
        if len(out) == n:
            break
    return out


def reason(need: str, c: CarView, d: CarDetails | None) -> str:
    """Why the car fits, in recorded figures only (said by the model in its own words)."""
    seats = _seats(d)
    if need in ("family", "trip") and seats:
        out = f"{seats} seats" + (f" and a {d.boot_litres} litre boot" if d.boot_litres else "")
        return out + (f", {c.km:,} kilometres, a {c.year}" if need == "trip" else "")
    if need == "sporty":
        if d and d.power_hp is not None:
            return f"{d.power_hp} horsepower" + (f", 0 to 100 in {d.accel_0_100:g} seconds" if d.accel_0_100 else "")
        return "a coupe"
    if need == "economy":
        cons = _consumption(d)
        return (f"{c.fuel}" + (f", uses {d.consumption}" if cons is not None else "")) if c.fuel in ("electric", "hybrid") else f"uses {d.consumption}"
    if need == "luxury":
        return f"a {c.year} {c.body_type} at {c.price:,} dirhams"
    if need == "budget":
        return f"{c.price:,} dirhams"
    return ""


def car_line(c: CarView, d: CarDetails | None, spec: str, why: str = "") -> str:
    """One car as true facts for the model: the name, colour, price, the spec sheet's figures (`spec`: engine._spec_facts) and the price against the market reference when it is lower."""
    below = ""
    if d and d.market_price and d.market_price > c.price:
        below = f", priced {d.market_price - c.price:,} dirhams under the market reference price"
    return f"- {T.car_name(c)}: {c.color}, {c.price:,} dirhams, {c.km:,} kilometres{spec}{below}" + (f". WHY IT FITS: {why}" if why else "")


GUIDE = {
    "family": "The customer wants a car for a family or for several people.",
    "trip": "The customer wants a car for long drives or trips.",
    "sporty": "The customer wants something sporty or fast.",
    "economy": "The customer wants a car that is cheap to run.",
    "luxury": "The customer wants something upmarket.",
    "budget": "The customer wants the lowest price.",
    "any": "The customer has not said what they need yet.",
}

STYLE = ("Use only the figures above. Never call a car popular, a best seller, great, perfect, reliable, comfortable or economical unless a figure above says so, and never mention a "
         "feature, seat count, space or economy that is not listed.")


def facts_for(need: str, said: str, lines: list[str]) -> list[str]:
    if need == "any" or not lines:
        return [f"The customer said: \"{said}\". {GUIDE['any']}",
                "Do NOT list cars yet. Ask ONE short question about what they will use the car for, how many people it is for, or their budget, and you may give at most ONE example from "
                "these cars in stock, giving its WHY IT FITS:", *lines[:1], STYLE]
    return [f"The customer said: \"{said}\". {GUIDE[need]}", "These cars in stock fit it best (true facts):", *lines,
            "Recommend at most TWO of them in one or two short sentences and give each one's WHY IT FITS in the same sentence (the customer needs the reason, so this is the one case to "
            "say a figure), then ask ONE short question that narrows it down: their budget, how many people, or what they will use it for. " + STYLE]


def resolve_need(model_need: str, words: str, earlier: list[str]) -> str:
    """The need to use: the model's, if the caller's words (or what they said just before) bear it out; else what the words say; else `any`. A model that names a need the caller
    never expressed ("I'm looking for a car" read as `family`) would skip the question the caller needs to be asked."""
    heard = [need_from_words(x) for x in [words, *reversed(earlier)]]
    said = next((n for n in heard if n != "any"), "any")
    if model_need in NEEDS and model_need != "any" and model_need == said:
        return model_need
    return said


def kind_of(c: CarView, d: CarDetails | None) -> str:
    """For the spread of `any`: which kind of fit this car was picked for (the cheapest, the roomiest or the most powerful)."""
    if d and d.seats and d.seats >= 6:
        return "family"
    if d and d.power_hp and d.power_hp >= 300:
        return "sporty"
    return "budget"
