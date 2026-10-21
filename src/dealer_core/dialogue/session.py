"""Per-call state. One Session per call; nothing is shared between calls."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime
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
    last_lead: str = ""                                      # the previous hybrid lead-in (never repeated)
    rng: random.Random = field(default_factory=lambda: random.Random(7))      # picks among approved wording variants
    last_variant: dict[str, int] = field(default_factory=dict)
    negotiation: dict[str, NegotiationState] = field(default_factory=dict)
    offered_slots: list[datetime] = field(default_factory=list)
    slot_day: date | None = None                             # the day being discussed ("Saturday afternoon"), so a bare time means that day
    draft_slot: datetime | None = None
    draft_name: str | None = None
    draft_phone: str | None = None
    contact_purpose: str = "booking"                         # "booking" | "send"
    booking: BookingView | None = None                       # the confirmed booking, if any
    rescheduling: str | None = None                          # booking id being moved
    pending: Pending | None = None
    oos_streak: int = 0
    ended: bool = False
    history: list[tuple[str, str]] = field(default_factory=list)      # ("customer" | "agent", what was said), this call only, in memory
    unanswered_q: str = ""                                   # a car question we had to ask "which one?" about, kept so "either of them" answers it
    last_focus: str = ""                                     # how the last search utterance was interpreted (see focus.py)
    handoff: Handoff | None = None
    misheard: int = 0

    def remember(self, who: str, text: str) -> None:
        """Keep the conversation so far (this call only; never written to a log). The reply writer and the intent call read it."""
        self.history.append((who, text))
        del self.history[:-16]

    def last_agent(self) -> str:
        return next((t for w, t in reversed(self.history) if w == "agent"), "")

    @property
    def focus(self) -> str:
        """What the call is about right now, in one readable line (shown in the debug trace)."""
        from .focus import describe

        if self.state in (State.SLOT, State.CONTACT, State.CONFIRM):
            what = "booking a test drive"
        elif self.state is State.NEGOTIATING:
            what = f"the price of car {self.car}"
        elif self.car:
            what = f"one car ({self.car})"
        elif self.shown:
            what = f"{len(self.shown)} cars just read out"
        else:
            what = "no car yet"
        return f"{what}; looking for: {describe(self.filters)}" + (f"; already shown: {len(self.excluded)}" if self.excluded else "")

