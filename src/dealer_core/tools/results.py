"""Typed results returned by the tools. Expected business failures are values (a `kind`), not exceptions, so the
state machine can branch on them. Nothing here carries floor_price: the LLM and the templates only ever see these."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum

from ..models import BookingStatus, CarView
from ..policy.negotiation import Decision, NegotiationState

VOICE_MAX_CARS = 3          # never read more than 3 cars aloud; ask a narrowing question first


@dataclass(frozen=True)
class SearchResult:
    total: int
    cars: tuple[CarView, ...]            # at most VOICE_MAX_CARS
    needs_narrowing: bool                # more matches than we may read aloud: ask a narrowing question first


@dataclass(frozen=True)
class SlotsResult:
    stock_id: str
    branch_id: str
    day: date
    slots: tuple[datetime, ...]          # branch-local, timezone-aware
    reason: str = ""                     # "branch_closed", "no_free_slots", "car_unavailable" ...


class BookingKind(str, Enum):
    BOOKED = "booked"
    RESCHEDULED = "rescheduled"
    CANCELLED = "cancelled"
    CAR_UNAVAILABLE = "car_unavailable"
    SLOT_TAKEN = "slot_taken"
    SLOT_INVALID = "slot_invalid"
    DUPLICATE = "duplicate"
    INVALID_PHONE = "invalid_phone"
    NOT_FOUND = "not_found"
    NOT_MODIFIABLE = "not_modifiable"
    STALE = "stale"


@dataclass(frozen=True)
class BookingView:
    booking_id: str
    stock_id: str
    car_name: str
    branch_name: str
    start: datetime                      # branch-local
    end: datetime
    customer_name: str
    customer_phone: str                  # E.164
    status: BookingStatus
    version: int


@dataclass(frozen=True)
class BookingOutcome:
    kind: BookingKind
    booking: BookingView | None = None
    reason: str = ""
    alternatives: tuple[datetime, ...] = ()        # next free slots, when a slot was refused
    similar_cars: tuple[CarView, ...] = ()         # suggestions, when the car was sold/reserved
    existing: BookingView | None = None            # the booking to reschedule, on DUPLICATE


@dataclass(frozen=True)
class Handoff:
    handoff_id: str
    reason: str
    stock_id: str | None
    offer: int | None
    note: str
    created_at: datetime


@dataclass(frozen=True)
class PriceOutcome:
    decision: Decision
    car: CarView
    state: NegotiationState
    handoff: Handoff | None = None     # filed automatically whenever the decision requires a human


@dataclass(frozen=True)
class SendResult:
    ok: bool
    channel: str
    sent: int = 0
    message_id: str = ""
    reason: str = ""
    stock_ids: tuple[str, ...] = field(default_factory=tuple)
