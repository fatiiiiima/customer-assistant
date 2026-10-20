"""Domain errors. Tools turn these into spoken-friendly outcomes; nothing here is shown to a caller verbatim."""
from __future__ import annotations


class DomainError(Exception):
    """Base class for expected business-rule failures (as opposed to bugs)."""


class CarNotFound(DomainError):
    pass


class CarUnavailable(DomainError):
    """The car is reserved or sold (it can change DURING a call, so this is re-checked at confirmation)."""

    def __init__(self, stock_id: str, status: str) -> None:
        super().__init__(f"{stock_id} is {status}")
        self.stock_id, self.status = stock_id, status


class SlotInvalid(DomainError):
    """Outside the branch's working hours, in the past, not on the slot grid, or too far ahead."""


class SlotTaken(DomainError):
    pass


class DuplicateBooking(DomainError):
    """The same customer already has an active booking for this car (reschedule it instead of creating another)."""

    def __init__(self, booking_id: str) -> None:
        super().__init__(f"customer already has booking {booking_id} for this car")
        self.booking_id = booking_id


class BookingNotFound(DomainError):
    pass


class NotModifiable(DomainError):
    """The booking is already cancelled or in the past."""


class StaleBooking(DomainError):
    """The booking changed since the caller last read it (version mismatch): re-read and re-confirm with the customer."""
