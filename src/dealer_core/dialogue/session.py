"""Per-call state. One Session per call; nothing is shared between calls."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..models import CarView
from ..policy.negotiation import NegotiationState
from ..tools.results import BookingView, Handoff
from .states import State


@dataclass
class Pending:
    """A yes/no question the agent just asked, so a bare 'yes' is understood in context."""

    kind: str          # handoff_offer | amount_check | offer_booking | move_existing
    value: Any = None


@dataclass
class Session:
    state: State = State.GREETING
    car: str | None = None                                   # stock id of the car in focus
    shown: list[CarView] = field(default_factory=list)       # cars read out last, for "the second one" / "other models"
    excluded: set[str] = field(default_factory=set)          # cars already shown, excluded from "other models" searches
    filters: dict = field(default_factory=dict)              # accumulated search filters
    narrow_asks: int = 0
    negotiation: dict[str, NegotiationState] = field(default_factory=dict)
    offered_slots: list[datetime] = field(default_factory=list)
    draft_slot: datetime | None = None
    draft_name: str | None = None
    draft_phone: str | None = None
    contact_purpose: str = "booking"                         # "booking" | "send"
    booking: BookingView | None = None                       # the confirmed booking, if any
    rescheduling: str | None = None                          # booking id being moved
    pending: Pending | None = None
    oos_streak: int = 0
    ended: bool = False
    handoff: Handoff | None = None
    misheard: int = 0
