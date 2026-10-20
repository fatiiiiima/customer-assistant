"""Tools-layer invariants (acceptance criterion: 100%, deterministic, no LLM)."""
import dataclasses
import json
import threading
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from dealer_core.models import CarStatus, SearchFilters
from dealer_core.policy.negotiation import DecisionKind
from dealer_core.tools.escalation import EscalationReason, MockEscalationSink
from dealer_core.tools.messaging import MockChannel
from dealer_core.tools.phones import normalize_phone, speak_phone
from dealer_core.tools.results import VOICE_MAX_CARS, BookingKind
from dealer_core.tools.service import DealerService

DUBAI = ZoneInfo("Asia/Dubai")
NOW = datetime(2030, 3, 4, 10, 0, tzinfo=DUBAI)        # a Monday, 10:00 in Dubai
WED = date(2030, 3, 6)
FRI = date(2030, 3, 8)
SUN = date(2030, 3, 10)


def at(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=DUBAI)


@pytest.fixture
def svc(cfg, inventory, bookings):
    return DealerService(cfg, inventory, bookings, {"whatsapp": MockChannel()}, MockEscalationSink(), clock=lambda: NOW.astimezone(timezone.utc))


def car_in(svc, branch):
    return svc.inventory.search(SearchFilters(branch_id=branch), 1)[0]


@pytest.fixture
def car(svc, cfg):
    return car_in(svc, cfg.branches[0].id)


# ---- slots ---------------------------------------------------------------------------------------------------------
def test_each_branch_uses_its_own_hours(svc, cfg):
    dxb, shj = (car_in(svc, b.id) for b in cfg.branches)
    d, s = svc.check_test_drive_slots(dxb.stock_id, WED), svc.check_test_drive_slots(shj.stock_id, WED)
    assert d.slots[0] == at(WED, 9) and d.slots[-1] == at(WED, 18, 30)           # DXB Wed 09-19
    assert s.slots[0] == at(WED, 10) and s.slots[-1] == at(WED, 17, 30)          # SHJ Wed 10-18
    assert all((b - a) == timedelta(minutes=30) for a, b in zip(d.slots, d.slots[1:]))


def test_no_weekend_is_assumed_closed_days_come_from_config(svc, cfg):
    dxb, shj = (car_in(svc, b.id) for b in cfg.branches)
    assert svc.check_test_drive_slots(dxb.stock_id, FRI).slots[0] == at(FRI, 14)      # DXB opens late on Friday
    assert svc.check_test_drive_slots(shj.stock_id, FRI).reason == "branch_closed"    # SHJ has no Friday hours in config
    assert svc.check_test_drive_slots(dxb.stock_id, SUN).reason == "branch_closed"


def test_minimum_lead_time_and_horizon(svc, car):
    today = svc.check_test_drive_slots(car.stock_id, NOW.date())
    assert today.slots and today.slots[0] >= NOW + timedelta(minutes=60)
    far = NOW.date() + timedelta(days=60)
    assert svc.check_test_drive_slots(car.stock_id, far).slots == ()


def test_slots_exclude_unavailable_cars_and_taken_times(svc, car):
    svc.book_test_drive(car.stock_id, "Ali", "0501234567", at(WED, 11))
    free = svc.check_test_drive_slots(car.stock_id, WED).slots
    assert at(WED, 11) not in free and at(WED, 11, 30) in free
    svc.inventory.mark_status(car.stock_id, CarStatus.RESERVED)
    assert svc.check_test_drive_slots(car.stock_id, WED).reason == "car_unavailable"


# ---- booking -------------------------------------------------------------------------------------------------------
def test_happy_path_booking_is_returned_in_branch_local_time(svc, car):
    out = svc.book_test_drive(car.stock_id, "Mohammed Al Mansoori", "050 123 4567", at(WED, 11))
    assert out.kind is BookingKind.BOOKED
    b = out.booking
    assert b.customer_phone == "+971501234567" and b.start == at(WED, 11) and b.end == at(WED, 11, 30)


@pytest.mark.parametrize("start,reason", [
    (lambda: at(WED, 11, 15), "outside_hours_or_off_grid"),
    (lambda: at(WED, 7), "outside_hours_or_off_grid"),
    (lambda: at(WED, 23), "outside_hours_or_off_grid"),
    (lambda: at(SUN, 11), "branch_closed"),
    (lambda: NOW + timedelta(minutes=10), "too_soon"),
    (lambda: NOW - timedelta(days=1), "too_soon"),
    (lambda: NOW + timedelta(days=90), "too_far_ahead"),
    (lambda: datetime(2030, 3, 6, 11), "naive_datetime"),
])
def test_invalid_slots_are_refused_with_alternatives(svc, car, start, reason):
    out = svc.book_test_drive(car.stock_id, "Ali", "0501234567", start())
    assert out.kind is BookingKind.SLOT_INVALID and out.reason == reason
    assert all(svc.slots.invalid_reason(car.branch_id, s) is None for s in out.alternatives)


def test_taken_slot_offers_free_alternatives(svc, car):
    svc.book_test_drive(car.stock_id, "Ali", "0501111111", at(WED, 11))
    out = svc.book_test_drive(car.stock_id, "Sara", "0502222222", at(WED, 11))
    assert out.kind is BookingKind.SLOT_TAKEN and 1 <= len(out.alternatives) <= 3 and at(WED, 11) not in out.alternatives


def test_car_sold_mid_call_is_caught_at_confirmation_and_alternatives_offered(svc, car):
    assert svc.car_status(car.stock_id) is CarStatus.AVAILABLE          # the agent reads the booking back...
    svc.inventory.mark_status(car.stock_id, CarStatus.SOLD)             # ...and the car is sold on another channel
    out = svc.book_test_drive(car.stock_id, "Ali", "0501234567", at(WED, 11))
    assert out.kind is BookingKind.CAR_UNAVAILABLE and out.reason == "sold" and out.booking is None
    assert all(c.status is CarStatus.AVAILABLE and c.stock_id != car.stock_id for c in out.similar_cars)
    assert svc.find_booking("0501234567", car.stock_id) is None


def test_bad_phone_or_name_is_refused_before_touching_the_database(svc, car):
    assert svc.book_test_drive(car.stock_id, "Ali", "call me", at(WED, 11)).kind is BookingKind.INVALID_PHONE
    assert svc.book_test_drive(car.stock_id, "  ", "0501234567", at(WED, 11)).kind is BookingKind.INVALID_PHONE


def test_second_booking_by_same_customer_for_same_car_points_to_the_existing_one(svc, car):
    first = svc.book_test_drive(car.stock_id, "Ali", "0501234567", at(WED, 11)).booking
    out = svc.book_test_drive(car.stock_id, "Ali", "0501234567", at(WED, 12))
    assert out.kind is BookingKind.DUPLICATE and out.existing.booking_id == first.booking_id


def test_reschedule_updates_in_place_and_never_duplicates(svc, car):
    first = svc.book_test_drive(car.stock_id, "Ali", "0501234567", at(WED, 11)).booking
    out = svc.reschedule_booking(first.booking_id, at(WED, 14), expected_version=first.version)
    assert out.kind is BookingKind.RESCHEDULED and out.booking.booking_id == first.booking_id and out.booking.start == at(WED, 14)
    assert out.booking.version == first.version + 1
    assert svc.find_booking("0501234567", car.stock_id).booking_id == first.booking_id
    assert at(WED, 11) in svc.check_test_drive_slots(car.stock_id, WED).slots            # old slot freed


def test_reschedule_failure_modes(svc, car):
    b = svc.book_test_drive(car.stock_id, "Ali", "0501234567", at(WED, 11)).booking
    svc.book_test_drive(car.stock_id, "Sara", "0502222222", at(WED, 15))
    assert svc.reschedule_booking(b.booking_id, at(WED, 15)).kind is BookingKind.SLOT_TAKEN
    assert svc.reschedule_booking(b.booking_id, at(SUN, 11)).kind is BookingKind.SLOT_INVALID
    assert svc.reschedule_booking("nope", at(WED, 14)).kind is BookingKind.NOT_FOUND
    assert svc.reschedule_booking(b.booking_id, at(WED, 14), expected_version=99).kind is BookingKind.STALE
    assert svc.find_booking("0501234567", car.stock_id).start == at(WED, 11)                # all refusals left it untouched


def test_cancel_is_idempotent_and_frees_the_slot_for_someone_else(svc, car):
    b = svc.book_test_drive(car.stock_id, "Ali", "0501234567", at(WED, 11)).booking
    assert svc.cancel_booking(b.booking_id).kind is BookingKind.CANCELLED
    assert svc.cancel_booking(b.booking_id).kind is BookingKind.CANCELLED
    assert svc.find_booking("0501234567", car.stock_id) is None
    assert svc.book_test_drive(car.stock_id, "Sara", "0502222222", at(WED, 11)).kind is BookingKind.BOOKED
    assert svc.cancel_booking("nope").kind is BookingKind.NOT_FOUND


def test_racing_callers_for_one_slot_yield_exactly_one_booking(svc, car):
    kinds, barrier = [], threading.Barrier(12)

    def go(i):
        barrier.wait()
        kinds.append(svc.book_test_drive(car.stock_id, f"C{i}", f"05010000{i:02d}", at(WED, 16)).kind)

    ts = [threading.Thread(target=go, args=(i,)) for i in range(12)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert kinds.count(BookingKind.BOOKED) == 1 and kinds.count(BookingKind.SLOT_TAKEN) == 11


# ---- search / voice rules --------------------------------------------------------------------------------------------
def test_never_more_than_three_cars_and_a_narrowing_flag(svc):
    broad = svc.search_cars(SearchFilters(), limit=50)
    assert len(broad.cars) == VOICE_MAX_CARS and broad.total > VOICE_MAX_CARS and broad.needs_narrowing
    narrow = svc.search_cars(SearchFilters(make="porsche"))
    assert narrow.total <= VOICE_MAX_CARS and not narrow.needs_narrowing


def test_search_returns_only_sellable_cars(svc):
    assert all(c.status is CarStatus.AVAILABLE for c in svc.search_cars(SearchFilters(), 3).cars)


# ---- price / escalation --------------------------------------------------------------------------------------------
def test_below_floor_haggling_files_a_handoff_and_never_a_deal(svc, car):
    state, kinds, last = None, [], None
    for offer in [car.price * 0.6] * 5:
        out = svc.propose_price(car.stock_id, int(offer), state)
        state, last = out.state, out
        kinds.append(out.decision.kind)
    assert DecisionKind.ACCEPT not in kinds and kinds[-1] is DecisionKind.ESCALATE
    assert last.handoff and last.handoff.reason == EscalationReason.BELOW_FLOOR.value
    assert svc.escalations.handoffs and svc.escalations.handoffs[-1].stock_id == car.stock_id


def test_manager_range_offer_files_a_price_approval_handoff(svc, car):
    state = None
    for offer in (car.price * 0.9, car.price * 0.92):
        out = svc.propose_price(car.stock_id, int(offer), state)
        state = out.state
    out = svc.propose_price(car.stock_id, car.floor_price + 1, state)
    assert out.decision.kind is DecisionKind.NEEDS_MANAGER and out.handoff.reason == EscalationReason.PRICE_APPROVAL.value


def test_accepted_offer_needs_no_handoff(svc, car):
    out = svc.propose_price(car.stock_id, car.price)
    assert out.decision.kind is DecisionKind.ACCEPT and out.handoff is None


def test_explicit_escalation_is_recorded(svc, car):
    h = svc.escalate_to_human(EscalationReason.ANGRY, car.stock_id, note="caller shouting")
    assert svc.escalations.handoffs == [h] and h.reason == "angry" and h.handoff_id.startswith("H")


# ---- messaging -----------------------------------------------------------------------------------------------------
def test_send_details_carries_specs_and_photos_not_floor_or_vin(svc, cfg):
    cars = svc.inventory.search(SearchFilters(), 8)
    res = svc.send_details("whatsapp", [c.stock_id for c in cars], "0501234567")
    assert res.ok and res.sent == 5                                          # capped
    sent = svc.channels["whatsapp"].sent[0]
    assert sent["to"] == "+971501234567" and "driving licence" in sent["text"] and sent["media"]
    assert "floor" not in sent["text"].lower() and all(c.vin not in sent["text"] for c in cars)


def test_send_details_refusals(svc):
    assert svc.send_details("telegram", ["S1042"], "0501234567").reason == "unknown_channel"
    assert svc.send_details("whatsapp", ["S1042"], "12").reason == "invalid_phone"
    assert svc.send_details("whatsapp", ["NOPE"], "0501234567").reason == "no_such_cars"


# ---- the floor never leaves the policy engine --------------------------------------------------------------------------
def test_no_tool_output_contains_a_floor_price_or_vin(svc, car):
    outs = [svc.search_cars(SearchFilters(), 3), svc.get_car(car.stock_id), svc.check_test_drive_slots(car.stock_id, WED),
            svc.book_test_drive(car.stock_id, "Ali", "0501234567", at(WED, 11)), svc.propose_price(car.stock_id, int(car.price * 0.5)),
            svc.propose_price(car.stock_id, int(car.price * 0.97))]

    def dump(o):
        if dataclasses.is_dataclass(o):
            return {f.name: dump(getattr(o, f.name)) for f in dataclasses.fields(o)}
        if hasattr(o, "model_dump"):
            return o.model_dump(mode="json")
        if isinstance(o, (list, tuple)):
            return [dump(x) for x in o]
        return o

    text = json.dumps([dump(o) for o in outs], default=str).lower()
    assert "floor" not in text and car.vin.lower() not in text


# ---- phones --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("0501234567", "+971501234567"), ("050 123 4567", "+971501234567"), ("+971 50 123 4567", "+971501234567"),
    ("00971501234567", "+971501234567"), ("501234567", "+971501234567"), ("04 321 1234", "+97143211234"),
    ("+442071234567", None), ("12345", None), ("0431234", None), ("04 321 123456", None), ("0301234567", None),
    ("call me", None), ("", None),
])
def test_normalize_phone(raw, expected):
    assert normalize_phone(raw) == expected


def test_speak_phone_groups_digits_for_the_ear():
    assert speak_phone("+971501234567") == "zero five zero, one two three, four five six seven"
