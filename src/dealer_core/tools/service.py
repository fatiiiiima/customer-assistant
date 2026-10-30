"""DealerService: the shared backend. The voice agent and (later) the website chat call exactly these functions.

Plain code over the repository interfaces: no LLM in here, and nothing returned from here contains a floor price.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Callable

from ..config import Config
from ..errors import (BookingNotFound, CarNotFound, CarUnavailable, DuplicateBooking, NotModifiable, SlotTaken,
                      StaleBooking)
from ..models import Booking, Car, CarStatus, CarView, NewBooking, SearchFilters
from ..policy.negotiation import DecisionKind, NegotiationEngine, NegotiationState
from ..repos.base import BookingRepository, InventoryRepository
from ..site.details import CarDetails, SiteInfo
from .escalation import EscalationReason, EscalationSink, make_handoff
from .messaging import MessageChannel, compose_details
from .phones import normalize_phone
from .results import (VOICE_MAX_CARS, BookingKind, BookingOutcome, BookingView, Handoff, PriceOutcome, SearchResult,
                      SendResult, SlotsResult)
from .slots import SlotService

SEND_MAX_CARS = 5


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DealerService:
    def __init__(self, cfg: Config, inventory: InventoryRepository, bookings: BookingRepository,
                 channels: dict[str, MessageChannel], escalations: EscalationSink,
                 clock: Callable[[], datetime] = utc_now) -> None:
        self.cfg, self.inventory, self.bookings = cfg, inventory, bookings
        self.channels, self.escalations, self.clock = channels, escalations, clock
        self.slots = SlotService(cfg, bookings, clock)
        self.negotiation = NegotiationEngine(cfg.negotiation)

    # ---- spec sheets and showroom: read from the database on every use (a price or a sale changes during a call) --------------------
    def details(self, stock_id: str | None) -> CarDetails | None:
        """The car's spec sheet; any field None is "not recorded". None when the car has none (or no car)."""
        return self.inventory.details(stock_id) if stock_id else None

    def details_for(self, stock_ids) -> dict[str, CarDetails]:
        return self.inventory.details_many([s for s in stock_ids if s])

    def showroom(self) -> SiteInfo:
        return self.inventory.showroom()

    # ---- search / read -----------------------------------------------------------------------------------------
    def search_cars(self, filters: SearchFilters, limit: int = VOICE_MAX_CARS) -> SearchResult:
        """Filtered SQL search (never RAG). At most VOICE_MAX_CARS come back; `needs_narrowing` says to ask a question first."""
        limit = min(limit, VOICE_MAX_CARS)
        total = self.inventory.count(filters)
        cars = tuple(c.view() for c in self.inventory.search(filters, limit))
        return SearchResult(total=total, cars=cars, needs_narrowing=total > VOICE_MAX_CARS)

    def get_car(self, stock_id: str) -> CarView | None:
        car = self.inventory.get(stock_id)
        return car.view() if car else None

    def car_status(self, stock_id: str) -> CarStatus | None:
        """Cheap re-check used right before reading a booking back or confirming it (stock changes during a call)."""
        car = self.inventory.get(stock_id)
        return car.status if car else None

    # ---- test drives -------------------------------------------------------------------------------------------
    def check_test_drive_slots(self, stock_id: str, day: date) -> SlotsResult:
        car = self.inventory.get(stock_id)
        if car is None:
            raise CarNotFound(stock_id)
        if car.status is not CarStatus.AVAILABLE:
            return SlotsResult(stock_id, car.branch_id, day, (), reason="car_unavailable")
        free = self.slots.free_slots(car, day)
        reason = "" if free else ("branch_closed" if not self.slots.grid(car.branch_id, day) else "no_free_slots")
        return SlotsResult(stock_id, car.branch_id, day, tuple(free), reason)

    def _view(self, b: Booking) -> BookingView:
        car = self.inventory.get(b.stock_id)
        tz = self.slots.tz(b.branch_id)
        return BookingView(booking_id=b.booking_id, stock_id=b.stock_id, car_name=car.view().name if car else b.stock_id,
                           branch_name=self.slots.branch(b.branch_id).name, start=b.slot_start.astimezone(tz),
                           end=b.slot_end.astimezone(tz), customer_name=b.customer_name, customer_phone=b.customer_phone,
                           status=b.status, version=b.version)

    def _similar(self, car: Car, k: int = 3) -> tuple[CarView, ...]:
        """Alternatives when a car is gone: same body type within +/-20% of its price, available."""
        f = SearchFilters(body_type=car.body_type, min_price=int(car.price * 0.8), max_price=int(car.price * 1.2))
        return tuple(c.view() for c in self.inventory.search(f, k + 1) if c.stock_id != car.stock_id)[:k]

    def similar_to(self, stock_id: str, k: int = 3) -> tuple[CarView, ...]:
        """Cars like this one (same body type, within +/-20% of its price): offered when a car is sold mid-call."""
        car = self.inventory.get(stock_id)
        return self._similar(car, k) if car else ()

    def book_test_drive(self, stock_id: str, customer_name: str, customer_phone: str, slot_start: datetime) -> BookingOutcome:
        phone = normalize_phone(customer_phone)
        if phone is None or not customer_name.strip():
            return BookingOutcome(BookingKind.INVALID_PHONE, reason="phone_or_name")
        car = self.inventory.get(stock_id)
        if car is None:
            raise CarNotFound(stock_id)
        reason = self.slots.invalid_reason(car.branch_id, slot_start)
        if reason:
            return BookingOutcome(BookingKind.SLOT_INVALID, reason=reason, alternatives=tuple(self.slots.alternatives(car, slot_start)))
        try:
            # The repository re-checks the car's status inside the same transaction that inserts the booking.
            booking = self.bookings.create(NewBooking(stock_id=stock_id, customer_name=customer_name.strip(),
                                                      customer_phone=phone, slot_start=slot_start), self.cfg.test_drive.slot_minutes)
        except CarUnavailable as e:
            return BookingOutcome(BookingKind.CAR_UNAVAILABLE, reason=e.status, similar_cars=self._similar(car))
        except SlotTaken:
            return BookingOutcome(BookingKind.SLOT_TAKEN, alternatives=tuple(self.slots.alternatives(car, slot_start)))
        except DuplicateBooking as e:
            existing = self.bookings.get(e.booking_id)
            return BookingOutcome(BookingKind.DUPLICATE, existing=self._view(existing) if existing else None)
        return BookingOutcome(BookingKind.BOOKED, booking=self._view(booking))

    def reschedule_booking(self, booking_id: str, new_start: datetime, expected_version: int | None = None) -> BookingOutcome:
        """Moves the SAME booking. Never creates a second one."""
        current = self.bookings.get(booking_id)
        if current is None:
            return BookingOutcome(BookingKind.NOT_FOUND)
        car = self.inventory.get(current.stock_id)
        reason = self.slots.invalid_reason(current.branch_id, new_start)
        if reason and car:
            return BookingOutcome(BookingKind.SLOT_INVALID, reason=reason, alternatives=tuple(self.slots.alternatives(car, new_start)),
                                  booking=self._view(current))
        try:
            moved = self.bookings.reschedule(booking_id, new_start, self.cfg.test_drive.slot_minutes, expected_version)
        except CarUnavailable as e:
            return BookingOutcome(BookingKind.CAR_UNAVAILABLE, reason=e.status, similar_cars=self._similar(car) if car else (),
                                  booking=self._view(current))
        except SlotTaken:
            return BookingOutcome(BookingKind.SLOT_TAKEN, alternatives=tuple(self.slots.alternatives(car, new_start)) if car else (),
                                  booking=self._view(current))
        except StaleBooking:
            return BookingOutcome(BookingKind.STALE, booking=self._view(self.bookings.get(booking_id)))
        except NotModifiable as e:
            return BookingOutcome(BookingKind.NOT_MODIFIABLE, reason=str(e), booking=self._view(current))
        except BookingNotFound:
            return BookingOutcome(BookingKind.NOT_FOUND)
        return BookingOutcome(BookingKind.RESCHEDULED, booking=self._view(moved))

    def cancel_booking(self, booking_id: str) -> BookingOutcome:
        try:
            return BookingOutcome(BookingKind.CANCELLED, booking=self._view(self.bookings.cancel(booking_id)))
        except BookingNotFound:
            return BookingOutcome(BookingKind.NOT_FOUND)

    def find_booking(self, customer_phone: str, stock_id: str) -> BookingView | None:
        phone = normalize_phone(customer_phone)
        b = self.bookings.find_active(phone, stock_id) if phone else None
        return self._view(b) if b else None

    # ---- price -------------------------------------------------------------------------------------------------
    def propose_price(self, stock_id: str, offer: int | None, state: NegotiationState | None = None) -> PriceOutcome:
        """The policy engine decides. When the decision needs a human, the handoff is filed here so it cannot be forgotten."""
        car = self.inventory.get(stock_id)
        if car is None:
            raise CarNotFound(stock_id)
        state = state or self.negotiation.initial_state(car)
        decision, new_state = self.negotiation.evaluate(car, offer, state)
        handoff = None
        if decision.kind is DecisionKind.ESCALATE:
            handoff = self.escalate_to_human(EscalationReason.BELOW_FLOOR, stock_id, offer, "offer below what a manager could approve")
        elif decision.kind is DecisionKind.NEEDS_MANAGER:
            handoff = self.escalate_to_human(EscalationReason.PRICE_APPROVAL, stock_id, decision.amount, "needs manager approval")
        return PriceOutcome(decision=decision, car=car.view(), state=new_state, handoff=handoff)

    # ---- messaging / escalation ----------------------------------------------------------------------------------
    def send_details(self, channel: str, stock_ids: list[str], customer_phone: str) -> SendResult:
        """Full specs, photos and links go out as a message; they are never read aloud."""
        if not self.cfg.send_details.enabled:
            return SendResult(False, channel, reason="disabled")
        ch = self.channels.get(channel)
        if ch is None:
            return SendResult(False, channel, reason="unknown_channel")
        phone = normalize_phone(customer_phone)
        if phone is None:
            return SendResult(False, channel, reason="invalid_phone")
        cars = [c.view() for sid in stock_ids[:SEND_MAX_CARS] if (c := self.inventory.get(sid))]
        if not cars:
            return SendResult(False, channel, reason="no_such_cars")
        note = "Please bring a valid driving licence for the test drive." if self.cfg.test_drive.licence_required else None
        text, media = compose_details(cars, note)
        mid = ch.send(phone, text, media)
        return SendResult(True, channel, sent=len(cars), message_id=mid, stock_ids=tuple(c.stock_id for c in cars))

    def escalate_to_human(self, reason: EscalationReason, stock_id: str | None = None, offer: int | None = None, note: str = "") -> Handoff:
        h = make_handoff(f"H{uuid.uuid4().hex[:6]}", reason, stock_id, offer, note, self.clock())
        self.escalations.file(h)
        return h
