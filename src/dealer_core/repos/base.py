"""Repository interfaces. The state machine, policy engine and tools depend ONLY on these Protocols.

A real database or the website's API replaces the SQLite mock by implementing them; nothing above this layer changes.
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..models import Booking, Car, CarStatus, NewBooking, SearchFilters
from ..site.details import CarDetails, SiteInfo


class InventoryRepository(Protocol):
    def search(self, filters: SearchFilters, limit: int = 10, offset: int = 0) -> list[Car]:
        """Matching cars, cheapest first. Unavailable cars are excluded unless the filters ask for them."""

    def count(self, filters: SearchFilters) -> int:
        """How many cars match (used to decide whether to ask a narrowing question before reading any aloud)."""

    def get(self, stock_id: str) -> Car | None: ...

    def vocabulary(self) -> dict[str, list[str]]:
        """{make: [models]} for cars that can be sold: builds the NLU make/model gazetteer from real stock."""

    def details(self, stock_id: str) -> CarDetails | None:
        """The car's spec sheet (any field None: not recorded), or None when there is none."""

    def details_many(self, stock_ids) -> dict[str, CarDetails]: ...

    def showroom(self) -> SiteInfo:
        """Hours, locations, phone, WhatsApp, email (SiteInfo() when none was imported)."""

    def version(self) -> int:
        """Goes up with every change to the stock (an import, a car sold): the website API's cache key."""

    def snapshot(self) -> dict[str, tuple[Car, CarDetails | None]]: ...

    def apply_sync(self, cars: list[Car], details: list[CarDetails], statuses: dict[str, CarStatus], showroom: SiteInfo | None,
                   meta: dict[str, str]) -> None:
        """One transaction (inventory/sync.py): upsert the cars and spec sheets, set the statuses, the showroom and the meta."""


class BookingRepository(Protocol):
    def create(self, new: NewBooking, slot_minutes: int) -> Booking:
        """Insert a booking at the car's branch. The implementation MUST, atomically with the insert, (1) re-check the
        car is still AVAILABLE (CarUnavailable), (2) refuse a taken slot (SlotTaken) and (3) refuse a second active
        booking by the same customer for the same car (DuplicateBooking). Races must never yield two bookings."""

    def get(self, booking_id: str) -> Booking | None: ...

    def find_active(self, customer_phone: str, stock_id: str) -> Booking | None: ...

    def reschedule(self, booking_id: str, new_start: datetime, slot_minutes: int,
                   expected_version: int | None = None) -> Booking:
        """Move an existing booking (same id; never creates a second one). Re-checks the car and the new slot."""

    def cancel(self, booking_id: str) -> Booking:
        """Idempotent: cancelling an already-cancelled booking returns it unchanged."""

    def booked_starts(self, branch_id: str, start: datetime, end: datetime) -> set[datetime]:
        """Start times of active bookings at a branch within [start, end)."""
