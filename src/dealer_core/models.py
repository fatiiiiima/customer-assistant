"""Domain models. `Car` is INTERNAL (it carries floor_price); `CarView` is what tools, templates and the LLM ever see."""
from __future__ import annotations

import re

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class CarStatus(str, Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    SOLD = "sold"


class BodyType(str, Enum):
    SUV = "suv"
    SEDAN = "sedan"
    HATCHBACK = "hatchback"
    COUPE = "coupe"
    PICKUP = "pickup"
    VAN = "van"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _SpecWords:
    """How the regional spec and the service history are said (both may be None: not recorded)."""
    @property
    def spec_name(self) -> str | None:
        """"GCC" (from "GCC", "GCC spec" or "GCC specs"); None: not recorded."""
        return re.sub(r"\s*specs?$", "", self.regional_spec, flags=re.I) or None if self.regional_spec else None

    def spec_said(self) -> str:
        return f"{self.spec_name} spec" if self.spec_name else "regional spec not recorded"

    def service_said(self) -> str:
        return {True: "a full service history", False: "no service history on file", None: "service history not recorded"}[self.service_history]


class Car(_SpecWords, _Frozen):
    """One unique unit. Placeholder fields until a real inventory schema exists (TO CONFIRM)."""

    stock_id: str
    vin: str
    make: str
    model: str
    year: int
    km: int
    price: int                      # AED list price
    floor_price: int                # AED. NEVER leaves the policy engine
    status: CarStatus
    branch_id: str
    body_type: BodyType
    fuel: str = "petrol"
    transmission: str = "automatic"
    color: str = "white"
    regional_spec: str | None = "GCC"      # None: not recorded (an imported stock list without the column)
    service_history: bool | None = True    # None: not recorded
    image_urls: tuple[str, ...] = ()

    def view(self) -> "CarView":
        return CarView(**self.model_dump(exclude={"floor_price", "vin"}))


class CarView(_SpecWords, _Frozen):
    """Customer-facing car. Deliberately has no floor_price (and no VIN): it cannot be read aloud by mistake."""

    stock_id: str
    make: str
    model: str
    year: int
    km: int
    price: int
    status: CarStatus
    branch_id: str
    body_type: BodyType
    fuel: str
    transmission: str
    color: str
    regional_spec: str | None
    service_history: bool | None
    image_urls: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"{self.year} {self.make} {self.model}"



class SearchFilters(_Frozen):
    max_price: int | None = None
    min_price: int | None = None
    make: str | None = None
    model: str | None = None
    body_type: BodyType | None = None
    color: str | None = None
    year_min: int | None = None
    max_km: int | None = None
    branch_id: str | None = None
    include_unavailable: bool = False   # default: only cars that can actually be sold
    exclude_stock_ids: tuple[str, ...] = ()   # cars the caller was already shown ("show me other models")
    fuel: str | None = None             # "petrol", "diesel", "hybrid", "electric"
    min_seats: int | None = None        # from the spec sheet ("seven seats"): a car whose seats are not recorded does not match
    min_power: int | None = None        # horsepower, from the spec sheet ("something sporty")
    sort: str = "price"                 # "price" (cheapest first) or "power" (most powerful first)


class BookingStatus(str, Enum):
    BOOKED = "booked"
    CANCELLED = "cancelled"


class NewBooking(_Frozen):
    stock_id: str
    customer_name: str = Field(min_length=1)
    customer_phone: str = Field(pattern=r"^\+?\d{8,15}$")   # digits only (optional +); read back before confirming
    slot_start: datetime                                      # timezone-aware


class Booking(_Frozen):
    booking_id: str
    stock_id: str
    branch_id: str
    customer_name: str
    customer_phone: str
    slot_start: datetime
    slot_end: datetime
    status: BookingStatus
    created_at: datetime
    updated_at: datetime
    version: int = 1                  # incremented on every change: lets callers detect "someone else changed it"
