"""Human handoff. The sink is a Protocol (in production: a CRM ticket or a live transfer); the mock just records."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol

from .results import Handoff


class EscalationReason(str, Enum):
    ANGRY = "angry"
    REQUESTED_HUMAN = "requested_human"
    OUT_OF_SCOPE = "out_of_scope"
    BELOW_FLOOR = "below_floor"              # price below what any manager could approve
    PRICE_APPROVAL = "price_approval"        # within a manager's range: needs their approval
    MISUNDERSTOOD = "misunderstood"          # repeated NLU failure
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    SYSTEM_FAULT = "system_fault"


class EscalationSink(Protocol):
    def file(self, handoff: Handoff) -> None: ...


@dataclass
class MockEscalationSink:
    handoffs: list[Handoff] = field(default_factory=list)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    def new_id(self) -> str:
        return f"H{next(self._ids):04d}"

    def file(self, handoff: Handoff) -> None:
        self.handoffs.append(handoff)


def make_handoff(sink_id: str, reason: EscalationReason, stock_id: str | None, offer: int | None, note: str, now: datetime) -> Handoff:
    return Handoff(handoff_id=sink_id, reason=reason.value, stock_id=stock_id, offer=offer, note=note, created_at=now)
