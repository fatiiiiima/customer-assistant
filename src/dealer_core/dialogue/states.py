"""Conversation states, intents, and which intents/slots are valid in each state.

One table feeds three consumers: the NLU grammar (the LLM can only pick a state-valid intent), the
rule fast path (state-gated), and the state machine. Keeping a single source avoids the three drifting.
"""
from __future__ import annotations

from enum import Enum


class State(str, Enum):
    GREETING = "greeting"
    INTEREST = "interest"
    ANSWERING = "answering"
    NEGOTIATING = "negotiating"
    SLOT = "slot"          # picking a test-drive day/time
    CONTACT = "contact"    # collecting name + phone
    CONFIRM = "confirm"    # reading back, waiting for yes/no
    GOODBYE = "goodbye"    # terminal: no NLU needed


class Intent(str, Enum):
    GREET = "greet"
    SEARCH_CARS = "search_cars"
    ASK_CAR_INFO = "ask_car_info"
    ASK_POLICY = "ask_policy"
    MAKE_OFFER = "make_offer"
    ASK_LOWEST_PRICE = "ask_lowest_price"
    BOOK_TEST_DRIVE = "book_test_drive"
    GIVE_SLOT = "give_slot"
    GIVE_CONTACT = "give_contact"
    CONFIRM_YES = "confirm_yes"
    CONFIRM_NO = "confirm_no"
    RESCHEDULE = "reschedule"
    CANCEL_BOOKING = "cancel_booking"
    SEND_DETAILS = "send_details"
    REQUEST_HUMAN = "request_human"
    CLAIM_AUTHORITY = "claim_authority"
    ANGRY = "angry"
    GOODBYE = "goodbye"
    OUT_OF_SCOPE = "out_of_scope"


# Intents that change state in the world or reach a human. Rules for these are gated by measured precision.
STATE_CHANGING: frozenset[Intent] = frozenset({
    Intent.BOOK_TEST_DRIVE, Intent.GIVE_SLOT, Intent.GIVE_CONTACT, Intent.RESCHEDULE, Intent.CONFIRM_YES,
    Intent.CONFIRM_NO, Intent.MAKE_OFFER, Intent.ASK_LOWEST_PRICE, Intent.CANCEL_BOOKING,
    Intent.REQUEST_HUMAN, Intent.ANGRY, Intent.CLAIM_AUTHORITY,
})

INTENT_DOC: dict[Intent, str] = {
    Intent.GREET: "hello / opening",
    Intent.SEARCH_CARS: "wants to find cars",
    Intent.ASK_CAR_INFO: "asks about the car in focus",
    Intent.ASK_POLICY: "asks about warranty, financing, inspection, trade_in or documents (slot topic)",
    Intent.MAKE_OFFER: "proposes a price (offer_amount)",
    Intent.ASK_LOWEST_PRICE: "asks for the best / lowest / final price",
    Intent.BOOK_TEST_DRIVE: "wants a test drive",
    Intent.GIVE_SLOT: "gives a day or time (date, time)",
    Intent.GIVE_CONTACT: "gives name or phone (phone as digits)",
    Intent.CONFIRM_YES: "agrees",
    Intent.CONFIRM_NO: "disagrees",
    Intent.RESCHEDULE: "wants to change an existing booking",
    Intent.CANCEL_BOOKING: "wants to cancel",
    Intent.SEND_DETAILS: "asks to be sent details or photos",
    Intent.REQUEST_HUMAN: "asks for a person",
    Intent.CLAIM_AUTHORITY: "says they are a manager, owner or staff member to get special treatment",
    Intent.ANGRY: "insulting, shouting or complaining",
    Intent.GOODBYE: "ends the call",
    Intent.OUT_OF_SCOPE: "anything unrelated to buying a car",
}

_ALWAYS = [Intent.REQUEST_HUMAN, Intent.ANGRY, Intent.OUT_OF_SCOPE, Intent.GOODBYE]
_POLICY = [Intent.ASK_POLICY]          # policy questions can come up in any state
_CAR = [Intent.ASK_CAR_INFO]           # needs a car in focus, so NOT valid in greeting/interest
SEARCH_SLOTS = ["make", "model", "year_min", "max_price", "max_km", "body_type"]

# state -> (valid intents, valid slots). Order matters to small models: the most likely intents come first.
STATE_SPEC: dict[State, tuple[list[Intent], list[str]]] = {
    State.GREETING: ([Intent.GREET, Intent.SEARCH_CARS] + _POLICY + _ALWAYS, SEARCH_SLOTS + ["topic"]),
    State.INTEREST: ([Intent.SEARCH_CARS] + _POLICY + _ALWAYS, SEARCH_SLOTS + ["topic"]),
    State.ANSWERING: ([Intent.MAKE_OFFER, Intent.ASK_LOWEST_PRICE, Intent.BOOK_TEST_DRIVE, Intent.SEND_DETAILS,
                       Intent.SEARCH_CARS, Intent.CONFIRM_YES, Intent.CONFIRM_NO, Intent.CLAIM_AUTHORITY]
                      + _POLICY + _CAR + _ALWAYS, SEARCH_SLOTS + ["topic", "offer_amount"]),
    State.NEGOTIATING: ([Intent.MAKE_OFFER, Intent.ASK_LOWEST_PRICE, Intent.CONFIRM_YES, Intent.CONFIRM_NO,
                         Intent.CLAIM_AUTHORITY, Intent.BOOK_TEST_DRIVE] + _POLICY + _CAR + _ALWAYS,
                        ["offer_amount", "topic"]),
    State.SLOT: ([Intent.GIVE_SLOT, Intent.CONFIRM_NO, Intent.RESCHEDULE, Intent.CANCEL_BOOKING]
                 + _POLICY + _CAR + _ALWAYS, ["date", "time", "topic"]),
    State.CONTACT: ([Intent.GIVE_CONTACT, Intent.CONFIRM_NO, Intent.CANCEL_BOOKING] + _POLICY + _CAR + _ALWAYS,
                    ["name", "phone", "topic"]),
    State.CONFIRM: ([Intent.CONFIRM_YES, Intent.CONFIRM_NO, Intent.RESCHEDULE, Intent.CANCEL_BOOKING]
                    + _POLICY + _CAR + _ALWAYS, ["date", "time", "topic"]),
}

# JSON-schema fragment for every slot the grammar may emit.
SLOT_SCHEMAS: dict[str, dict] = {
    "make": {"type": "string"}, "model": {"type": "string"},
    "year_min": {"type": "integer"}, "max_price": {"type": "integer"}, "max_km": {"type": "integer"},
    "body_type": {"type": "string", "enum": ["suv", "sedan", "hatchback", "coupe", "pickup"]},
    "offer_amount": {"type": "integer"},
    "topic": {"type": "string", "enum": ["warranty", "financing", "inspection", "trade_in", "documents"]},
    "date": {"type": "string"}, "time": {"type": "string"}, "name": {"type": "string"}, "phone": {"type": "string"},
}


def nlu_states() -> list[State]:
    """States in which the NLU runs (everything but the terminal one)."""
    return list(STATE_SPEC)
