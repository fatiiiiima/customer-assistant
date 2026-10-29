"""How the code-written sentences are said: short, spoken English, with a few variants each so a call does not sound like a form.

Every variant says exactly the same facts (the facts are filled in by code); only the wording around them changes. The variant is picked with the
session's own random generator, never the same one twice in a row for the same kind of sentence.
"""
from __future__ import annotations

import re
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
    tail = "" if more <= 0 else f", and {more} more"                 # (one or two more are named instead: engine._t_search_cars, results.NAME_THE_REST_UP_TO)
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


OPENERS = ["Sure,", "Okay,", "Right,", "Alright,", "Let's see,", "So,"]
NEUTRAL_OPENERS = ["Okay,", "Right,", "Hmm,"]                  # before a "no" ('Sure, I don't see a sunroof' sounds wrong)


def checking(s: Session) -> str:
    """Said before the free times: the voice has it ready, so it plays while the times are being spoken into audio."""
    return pick(s, "checking", ["Let me check.", "Let me see.", "Let me have a look."])


def opener(s: Session, negative: bool = False) -> str:
    """A short acknowledgement before an answer (not the same one twice running)."""
    return pick(s, "opener_no" if negative else "opener", NEUTRAL_OPENERS if negative else OPENERS)


def anything_else(s: Session) -> str:
    return pick(s, "else", ["Is there anything else I can help with?", "What else can I do for you?", "Anything else you'd like to know?"])


# What counts as an invitation to visit (kept equal to engine._OFFERED_DRIVE, which recognises the caller's "yes" to one)
INVITES = re.compile(r"\b(?:test drive|take a look|have a look|look at it|check it out|come (?:in|and see|by)|see it|in person|firsthand|behind the wheel|take (?:it|her) for a spin|a spin|"
                     r"drive it|schedule|book|set up|visit|viewing|try it|experience it)\b")
# an invitation at most every third answer (27 Oct: four replies in a row ended with a different invitation to visit, after the caller had said "No")
INVITE_EVERY = 6          # history entries: three exchanges


def offer_test_drive(s: Session) -> str:
    """The closing question of an answer about a car: an invitation to visit now and then, otherwise a plain \"anything else about it?\".
    Not while the caller has just said no (Session.no_invite_until), and never in two answers running."""
    if len(s.history) >= s.no_invite_until:
        s.no_invite_until = len(s.history) + INVITE_EVERY
        return pick(s, "offer", ["Would you like to come and see it?", "Want to book a test drive?", "Shall I set up a test drive for you?"])
    return pick(s, "more_about", ["Anything else you'd like to know about it?", "What else would you like to know about it?", "Is there anything else you'd like to hear about it?"])


def hear_more_or_book(s: Session) -> str:
    """After a search that found exactly one car."""
    if len(s.history) >= s.no_invite_until:
        s.no_invite_until = len(s.history) + INVITE_EVERY
        return "Would you like to hear more, or book a test drive?"
    return "Would you like to hear more about it?"


def nothing_more(s: Session) -> str:
    """Every fact on the car's sheet has been told: say so, instead of telling one again."""
    return pick(s, "nothing_more", ["That's everything I have on it. Anything else I can help with?", "I've told you all I have on that one. Would you like to see another car?",
                                    "That's all I've got on it. Is there anything else you'd like to know?"])


def ask_day(s: Session, c: CarView) -> str:
    return pick(s, "ask_day", ["Sure! What day would you like to come in?", f"Great choice. Which day works for you to try the {c.model}?",
                               "Of course. What day and time suit you?"])


def offer_slot_variants(day_said: str, times: str) -> list[str]:
    cap = day_said[:1].upper() + day_said[1:]
    return [f"{cap} I can do {times}. Which suits you better?", f"{cap} I've got {times}. Which works for you?",
            f"The soonest is {day_said}: {times}. Which would you prefer?"]


def offer_slots(s: Session, day_said: str, times: str, intro: str = "") -> str:
    """Real times offered instead of a question: "Tomorrow I can do 9 in the morning or 2 in the afternoon. Which suits you better?" `intro` (e.g. "There's nothing free today.") comes first."""
    body = pick(s, "offer_slots", offer_slot_variants(day_said, times))
    return f"{intro} {body}" if intro else body


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
