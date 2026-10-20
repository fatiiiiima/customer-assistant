import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from dealer_core.errors import (BookingNotFound, CarNotFound, CarUnavailable, DuplicateBooking, NotModifiable, SlotTaken,
                                StaleBooking)
from dealer_core.models import BookingStatus, BodyType, CarStatus, NewBooking, SearchFilters

SLOT = 30
T0 = datetime(2030, 3, 4, 10, 0, tzinfo=timezone.utc)      # far in the future: these tests check storage, not opening hours


def new(stock="S1042", phone="0501234567", start=T0, name="Test Customer"):
    return NewBooking(stock_id=stock, customer_name=name, customer_phone=phone, slot_start=start)


# ---- inventory ---------------------------------------------------------------------------------------------------
def test_seed_has_sixty_cars_two_branches_and_all_statuses(inventory, cfg):
    allc = inventory.search(SearchFilters(include_unavailable=True), limit=100)
    assert len(allc) == 60
    assert {c.branch_id for c in allc} == {b.id for b in cfg.branches}
    assert {c.status for c in allc} == set(CarStatus)
    assert all(c.floor_price <= c.price for c in allc)


def test_search_defaults_to_available_only_and_is_cheapest_first(inventory):
    cars = inventory.search(SearchFilters(), limit=100)
    assert cars and all(c.status is CarStatus.AVAILABLE for c in cars)
    assert [c.price for c in cars] == sorted(c.price for c in cars)


def test_search_filters(inventory):
    f = SearchFilters(body_type=BodyType.SUV, max_price=150_000, year_min=2018, max_km=120_000)
    for c in inventory.search(f, limit=100):
        assert c.body_type is BodyType.SUV and c.price <= 150_000 and c.year >= 2018 and c.km <= 120_000
    assert inventory.count(f) == len(inventory.search(f, limit=100))


def test_search_make_is_case_insensitive(inventory):
    a = inventory.search(SearchFilters(make="toyota"), limit=100)
    b = inventory.search(SearchFilters(make="TOYOTA"), limit=100)
    assert a and [c.stock_id for c in a] == [c.stock_id for c in b]


def test_get_unknown_returns_none(inventory):
    assert inventory.get("NOPE") is None


def test_vocabulary_lists_only_sellable_cars(inventory):
    vocab = inventory.vocabulary()
    assert "toyota" in vocab and all(isinstance(v, list) and v for v in vocab.values())


def test_public_view_never_contains_floor_price_or_vin(inventory):
    car = inventory.get("S1042")
    view = car.view()
    dumped = view.model_dump_json()
    assert "floor_price" not in dumped and "vin" not in dumped and car.vin not in dumped
    assert not hasattr(view, "floor_price") and not hasattr(view, "vin")


def test_db_rejects_floor_above_price(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute("INSERT INTO cars VALUES ('X','v','m','m',2020,1,100,200,'available','b','suv','p','a','c','GCC',1,'[]')")


# ---- bookings ----------------------------------------------------------------------------------------------------
def test_create_and_read_back(bookings):
    b = bookings.create(new(), SLOT)
    assert b.status is BookingStatus.BOOKED and b.slot_end - b.slot_start == timedelta(minutes=SLOT) and b.version == 1
    assert bookings.get(b.booking_id) == b
    assert bookings.find_active("0501234567", "S1042") == b


def test_same_slot_same_car_is_refused(bookings):
    bookings.create(new(phone="0501111111"), SLOT)
    with pytest.raises(SlotTaken):
        bookings.create(new(phone="0502222222"), SLOT)


def test_same_customer_same_car_is_a_duplicate_and_points_to_the_existing_booking(bookings):
    first = bookings.create(new(), SLOT)
    with pytest.raises(DuplicateBooking) as e:
        bookings.create(new(start=T0 + timedelta(hours=1)), SLOT)
    assert e.value.booking_id == first.booking_id


def test_customer_cannot_be_in_two_places_at_once(bookings, inventory):
    other = next(c.stock_id for c in inventory.search(SearchFilters(), 100) if c.stock_id != "S1042")
    bookings.create(new(), SLOT)
    with pytest.raises(SlotTaken):
        bookings.create(new(stock=other), SLOT)


def test_unavailable_or_unknown_car_is_refused(bookings, inventory):
    inventory.mark_status("S1042", CarStatus.SOLD)
    with pytest.raises(CarUnavailable) as e:
        bookings.create(new(), SLOT)
    assert e.value.status == "sold"
    with pytest.raises(CarNotFound):
        bookings.create(new(stock="NOPE"), SLOT)


def test_status_is_rechecked_when_rescheduling_too(bookings, inventory):
    b = bookings.create(new(), SLOT)
    inventory.mark_status("S1042", CarStatus.RESERVED)          # sold/reserved by another channel mid-call
    with pytest.raises(CarUnavailable):
        bookings.reschedule(b.booking_id, T0 + timedelta(hours=2), SLOT)


def test_reschedule_updates_the_same_booking_and_bumps_version(bookings):
    b = bookings.create(new(), SLOT)
    moved = bookings.reschedule(b.booking_id, T0 + timedelta(hours=3), SLOT, expected_version=1)
    assert moved.booking_id == b.booking_id and moved.version == 2 and moved.slot_start == T0 + timedelta(hours=3)
    assert bookings.find_active("0501234567", "S1042").booking_id == b.booking_id        # still exactly one
    with pytest.raises(StaleBooking):
        bookings.reschedule(b.booking_id, T0 + timedelta(hours=4), SLOT, expected_version=1)


def test_reschedule_into_a_taken_slot_is_refused_and_leaves_the_booking_untouched(bookings):
    bookings.create(new(phone="0509999999", start=T0 + timedelta(hours=1)), SLOT)      # someone else, same car, 11:00
    mine = bookings.create(new(), SLOT)                                                 # me, same car, 10:00
    with pytest.raises(SlotTaken):
        bookings.reschedule(mine.booking_id, T0 + timedelta(hours=1), SLOT)
    after = bookings.get(mine.booking_id)
    assert after.slot_start == T0 and after.version == mine.version                    # nothing changed


def test_cancel_is_idempotent_and_frees_the_slot(bookings):
    b = bookings.create(new(), SLOT)
    c1 = bookings.cancel(b.booking_id)
    c2 = bookings.cancel(b.booking_id)
    assert c1.status is BookingStatus.CANCELLED and c2.version == c1.version            # second cancel changes nothing
    assert bookings.create(new(phone="0507777777"), SLOT).status is BookingStatus.BOOKED   # slot reusable
    with pytest.raises(NotModifiable):
        bookings.reschedule(b.booking_id, T0 + timedelta(hours=5), SLOT)


def test_unknown_booking(bookings):
    with pytest.raises(BookingNotFound):
        bookings.cancel("nope")


def test_booked_starts_range(bookings):
    b = bookings.create(new(), SLOT)
    branch = b.branch_id
    assert bookings.booked_starts(branch, T0 - timedelta(hours=1), T0 + timedelta(hours=1)) == {T0}
    assert bookings.booked_starts(branch, T0 + timedelta(hours=1), T0 + timedelta(hours=2)) == set()
    assert bookings.booked_starts("other-branch", T0 - timedelta(hours=1), T0 + timedelta(hours=1)) == set()


def test_database_guards_hold_even_if_python_checks_are_bypassed(db, bookings):
    """The unique indexes are the real defence: a raw INSERT that skips every Python check must still fail."""
    b = bookings.create(new(), SLOT)
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute("INSERT INTO bookings VALUES ('raw1',?,?, 'Eve','0508888888',?,?,'booked',0,0,1)",
                        (b.stock_id, b.branch_id, int(b.slot_start.timestamp()), int(b.slot_end.timestamp())))
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute("INSERT INTO bookings VALUES ('raw2',?,?, 'Bob','0501234567',?,?,'booked',0,0,1)",
                        (b.stock_id, b.branch_id, int(b.slot_start.timestamp()) + 7200, int(b.slot_end.timestamp()) + 7200))


def test_many_threads_racing_for_one_slot_produce_exactly_one_booking(bookings):
    results, barrier = [], threading.Barrier(16)

    def worker(i):
        barrier.wait()
        try:
            bookings.create(new(phone=f"05000000{i:02d}"), SLOT)
            results.append("ok")
        except SlotTaken:
            results.append("taken")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count("ok") == 1 and results.count("taken") == 15


def test_phone_must_be_digits(bookings):
    with pytest.raises(Exception):
        new(phone="call me maybe")
