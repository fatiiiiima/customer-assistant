"""How the code-written sentences are said: short, spoken English, with a few variants each so a call does not sound like a form.

Every variant says exactly the same facts (the facts are filled in by code); only the wording around them changes. The variant is picked with the
session's own random generator, never the same one twice in a row for the same kind of sentence.
"""
from __future__ import annotations

from datetime import datetime

from ..dialogue import templates as T
from ..dialogue.session import Session
from ..models import CarView


def pick(s: Session, kind: str, options: list[str]) -> str:
    """One of `options`, not the one used last time for this `kind`."""
    last = s.last_variant.get(kind, -1)
    choices = [i for i in range(len(options)) if i != last] or [0]
    i = s.rng.choice(choices)
    s.last_variant[kind] = i
    return options[i]


def plural(word: str) -> str:
    return word if word.endswith("s") else word + "s"


def what_they_asked(cars: list[CarView], filters: dict) -> str:
    """'Toyotas', 'SUVs', 'Toyota SUVs', 'cars': what the matches have in common, the way a person would say it."""
    make = filters.get("make")
    body = filters.get("body_type")
    model = filters.get("model")
    if model and cars:
        return plural(T.short_name(cars[0]))
    name = next((c.make for c in cars), make.title()) if make else ""
    kind = {"suv": "SUVs", "sedan": "sedans", "hatchback": "hatchbacks", "coupe": "coupes", "van": "vans", "pickup": "pickups"}.get(str(body or ""), "")
    colour = str(filters.get("color") or "")
    if name and kind:
        return f"{colour + ' ' if colour else ''}{name} {kind}"
    if colour:
        return f"{colour} {kind or (plural(name) if name else 'cars')}"
    fuel = filters.get("fuel")
    if fuel:
        return f"{'electric' if fuel == 'electric' else fuel} {kind or 'cars'}"
    return plural(name) if name else kind or "cars that fit"


def count_line(s: Session, total: int, cars: list[CarView], filters: dict) -> str:
    what = what_they_asked(cars, filters)
    if filters.get("seats"):
        what = "cars with seven seats or more"
    elif filters.get("sporty"):
        what = "sporty cars"
    elif what == "cars that fit" and not any(filters.values()):
        what = "cars"
    return pick(s, "count", [f"We've got {total} {what} in at the moment.", f"There are {total} {what} in stock right now.", f"I've got {total} {what} for you."])


def list_line(s: Session, cars: list[CarView], total: int) -> str:
    """'There's the Corolla at 62,500 dirhams, the Camry at 74,500 and the RAV4 at 98,500, plus one more.' Dirhams said once."""
    same_make = len({c.make for c in cars}) == 1
    name = (lambda c: T.short_name(c)) if same_make else (lambda c: f"{c.make} {c.model}" if not T.short_name(c).startswith(c.make) else T.short_name(c))
    parts = [f"the {name(c)} at {T.money(c.price) if i == 0 else f'{c.price:,}'}" for i, c in enumerate(cars)]
    listed = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    more = total - len(cars)
    tail = "" if more <= 0 else (", plus one more" if more == 1 else f", plus {more} more")
    if len(cars) == 1:
        return pick(s, "one", [f"I've got {listed}.", f"There's {listed}."])
    return pick(s, "list", [f"There's {listed}{tail}.", f"I've got {listed}{tail}.", f"We have {listed}{tail}."])


def car_line(s: Session, c: CarView) -> str:
    km, price = f"{c.km:,} kilometres", T.money(c.price)
    return pick(s, "car", [
        f"It's the {c.year} {c.make} {c.model} in {c.color}, with {km} on it, at {price}.",
        f"That's a {c.year} {c.model}, {c.color}, {km}, and it's {price}.",
        f"The {c.model} is a {c.year} in {c.color}. It's done {km} and it's {price}.",
    ])


def anything_else(s: Session) -> str:
    return pick(s, "else", ["Is there anything else I can help with?", "What else can I do for you?", "Anything else you'd like to know?"])


def offer_test_drive(s: Session) -> str:
    return pick(s, "offer", ["Would you like to come and see it?", "Want to book a test drive?", "Shall I set up a test drive for you?"])


def ask_day(s: Session, c: CarView) -> str:
    return pick(s, "ask_day", ["Sure! What day would you like to come in?", f"Great choice. Which day works for you to try the {c.model}?",
                               "Of course. What day and time suit you?"])


def ask_missing(s: Session, missing: list[str], name: str | None, time_said: str = "") -> str:
    """The next detail the booking needs, asked the way a person would. `time_said`: the slot the caller just gave, said back first
    ("Great, tomorrow at 5 PM.") so a misheard time is caught at once, not only at the read-back."""
    if time_said:
        ask = {("mobile number",): "And your mobile number?", ("name",): "And your name, please?"}.get(tuple(missing), "Can I take your name and mobile number?")
        return pick(s, "ack_time", [f"Great, {time_said} it is. {ask}", f"Lovely, {time_said}. {ask}", f"Perfect, {time_said}. {ask}"])
    thanks = f"Thanks, {name.split()[0]}." if name and "name" not in missing else "Thanks."
    if missing == ["mobile number"]:
        return pick(s, "ask_phone", [f"{thanks} And your mobile number?", f"{thanks} What's the best mobile number to reach you on?"])
    if missing == ["name"]:
        return pick(s, "ask_name", ["Great. And your name, please?", "Lovely. Who should I put the booking under?"])
    if set(missing) == {"name", "mobile number"}:
        return pick(s, "ask_contact", ["Perfect. Can I take your name and mobile number?", "Great. What's your name and mobile number?"])
    if missing[0] == "day and time":
        return pick(s, "ask_when", ["What day and time work for you?", "When would you like to come in?"])
    return f"Could I have your {' and '.join(missing)}?"


def read_back(c: CarView, when: datetime, branch: str, name: str, phone_spoken: str) -> str:
    """All five details, once, then the question. (Kept to one wording: it is the check before anything is booked.)"""
    return f"So that's the {c.year} {c.model} on {T.when_full(when)}, at our {branch}, for {name}, on {phone_spoken}. Shall I book it?"


def booked(s: Session, when_said: str, licence: bool) -> str:
    lic = " Just bring your driving licence along." if licence else ""
    return pick(s, "booked", [f"Done, you're booked for {when_said}.{lic}", f"All set, you're booked in for {when_said}.{lic}"])
