"""SQLite MOCK implementation of the inventory and booking repositories (fake data only).

Safety is enforced in the database, not just in Python: partial UNIQUE indexes make a double booking impossible even if
two writers race, and every write runs inside BEGIN IMMEDIATE so the car-status re-check and the insert are atomic.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterator

from ..errors import (BookingNotFound, CarNotFound, CarUnavailable, DuplicateBooking, NotModifiable, SlotTaken,
                      StaleBooking)
from ..models import Booking, BookingStatus, BodyType, Car, CarStatus, NewBooking, SearchFilters

SCHEMA = """
CREATE TABLE IF NOT EXISTS cars (
  stock_id TEXT PRIMARY KEY, vin TEXT NOT NULL, make TEXT NOT NULL, model TEXT NOT NULL, year INTEGER NOT NULL,
  km INTEGER NOT NULL, price INTEGER NOT NULL, floor_price INTEGER NOT NULL, status TEXT NOT NULL,
  branch_id TEXT NOT NULL, body_type TEXT NOT NULL, fuel TEXT NOT NULL, transmission TEXT NOT NULL, color TEXT NOT NULL,
  regional_spec TEXT NOT NULL, service_history INTEGER NOT NULL, image_urls TEXT NOT NULL,
  CHECK (floor_price <= price AND floor_price > 0)
);
CREATE INDEX IF NOT EXISTS ix_cars_search ON cars(status, make, model, price);
CREATE TABLE IF NOT EXISTS bookings (
  booking_id TEXT PRIMARY KEY, stock_id TEXT NOT NULL REFERENCES cars(stock_id), branch_id TEXT NOT NULL,
  customer_name TEXT NOT NULL, customer_phone TEXT NOT NULL, slot_start INTEGER NOT NULL, slot_end INTEGER NOT NULL,
  status TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, version INTEGER NOT NULL
);
-- The three double-booking guards. Cancelled rows are exempt so a slot can be reused.
CREATE UNIQUE INDEX IF NOT EXISTS ux_car_slot      ON bookings(stock_id, slot_start)       WHERE status = 'booked';
CREATE UNIQUE INDEX IF NOT EXISTS ux_customer_car  ON bookings(customer_phone, stock_id)   WHERE status = 'booked';
CREATE UNIQUE INDEX IF NOT EXISTS ux_customer_slot ON bookings(customer_phone, slot_start) WHERE status = 'booked';
"""


def _ts(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware")
    return int(dt.timestamp())


def _dt(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


class SqliteDb:
    """One connection guarded by a lock (sqlite3 connections are not thread-safe); writers use BEGIN IMMEDIATE."""

    def __init__(self, path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.lock = threading.RLock()
        self.conn.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    def query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, args).fetchall()


def _car(r: sqlite3.Row) -> Car:
    return Car(stock_id=r["stock_id"], vin=r["vin"], make=r["make"], model=r["model"], year=r["year"], km=r["km"],
               price=r["price"], floor_price=r["floor_price"], status=CarStatus(r["status"]), branch_id=r["branch_id"],
               body_type=BodyType(r["body_type"]), fuel=r["fuel"], transmission=r["transmission"], color=r["color"],
               regional_spec=r["regional_spec"], service_history=bool(r["service_history"]),
               image_urls=tuple(json.loads(r["image_urls"])))


class SqliteInventoryRepo:
    def __init__(self, db: SqliteDb) -> None:
        self.db = db

    def add(self, car: Car) -> None:
        """Seeding helper (not part of the Protocol)."""
        with self.db.tx() as c:
            c.execute("INSERT INTO cars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (car.stock_id, car.vin, car.make, car.model, car.year, car.km, car.price, car.floor_price,
                       car.status.value, car.branch_id, car.body_type.value, car.fuel, car.transmission, car.color,
                       car.regional_spec, int(car.service_history), json.dumps(list(car.image_urls))))

    def mark_status(self, stock_id: str, status: CarStatus) -> None:
        """Simulates ANOTHER channel selling/reserving a car mid-call (mock only; not part of the Protocol)."""
        with self.db.tx() as c:
            if c.execute("UPDATE cars SET status = ? WHERE stock_id = ?", (status.value, stock_id)).rowcount == 0:
                raise CarNotFound(stock_id)

    @staticmethod
    def _where(f: SearchFilters) -> tuple[str, list]:
        where, args = [], []
        if not f.include_unavailable:
            where.append("status = 'available'")
        for col, val, op in (("price", f.max_price, "<="), ("price", f.min_price, ">="), ("year", f.year_min, ">="),
                             ("km", f.max_km, "<="), ("branch_id", f.branch_id, "=")):
            if val is not None:
                where.append(f"{col} {op} ?")
                args.append(val)
        for col, val in (("make", f.make), ("model", f.model), ("color", f.color)):
            if val:
                where.append(f"LOWER({col}) = ?")
                args.append(val.strip().lower())
        if f.body_type:
            where.append("body_type = ?")
            args.append(f.body_type.value)
        if f.exclude_stock_ids:
            where.append("stock_id NOT IN (%s)" % ",".join("?" * len(f.exclude_stock_ids)))
            args.extend(f.exclude_stock_ids)
        return (" WHERE " + " AND ".join(where)) if where else "", args

    def search(self, filters: SearchFilters, limit: int = 10) -> list[Car]:
        w, args = self._where(filters)
        return [_car(r) for r in self.db.query(f"SELECT * FROM cars{w} ORDER BY price, stock_id LIMIT ?", (*args, limit))]

    def count(self, filters: SearchFilters) -> int:
        w, args = self._where(filters)
        return self.db.query(f"SELECT COUNT(*) AS n FROM cars{w}", tuple(args))[0]["n"]

    def get(self, stock_id: str) -> Car | None:
        rows = self.db.query("SELECT * FROM cars WHERE stock_id = ?", (stock_id,))
        return _car(rows[0]) if rows else None

    def vocabulary(self) -> dict[str, list[str]]:
        vocab: dict[str, list[str]] = {}
        for r in self.db.query("SELECT DISTINCT make, model FROM cars WHERE status = 'available' ORDER BY make, model"):
            vocab.setdefault(r["make"].lower(), []).append(r["model"].lower())
        return vocab


def _booking(r: sqlite3.Row) -> Booking:
    return Booking(booking_id=r["booking_id"], stock_id=r["stock_id"], branch_id=r["branch_id"],
                   customer_name=r["customer_name"], customer_phone=r["customer_phone"], slot_start=_dt(r["slot_start"]),
                   slot_end=_dt(r["slot_end"]), status=BookingStatus(r["status"]), created_at=_dt(r["created_at"]),
                   updated_at=_dt(r["updated_at"]), version=r["version"])


class SqliteBookingRepo:
    def __init__(self, db: SqliteDb) -> None:
        self.db = db

    @staticmethod
    def _require_available(c: sqlite3.Connection, stock_id: str) -> sqlite3.Row:
        row = c.execute("SELECT stock_id, branch_id, status FROM cars WHERE stock_id = ?", (stock_id,)).fetchone()
        if row is None:
            raise CarNotFound(stock_id)
        if row["status"] != CarStatus.AVAILABLE.value:
            raise CarUnavailable(stock_id, row["status"])
        return row

    def create(self, new: NewBooking, slot_minutes: int) -> Booking:
        start = _ts(new.slot_start)
        now = _ts(datetime.now(timezone.utc))
        bid = uuid.uuid4().hex[:12]
        with self.db.tx() as c:
            car = self._require_available(c, new.stock_id)          # re-checked INSIDE the transaction that inserts
            dup = c.execute("SELECT booking_id FROM bookings WHERE customer_phone = ? AND stock_id = ? AND status = 'booked'",
                            (new.customer_phone, new.stock_id)).fetchone()
            if dup:
                raise DuplicateBooking(dup["booking_id"])
            try:
                c.execute("INSERT INTO bookings VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                          (bid, new.stock_id, car["branch_id"], new.customer_name, new.customer_phone, start,
                           start + slot_minutes * 60, BookingStatus.BOOKED.value, now, now, 1))
            except sqlite3.IntegrityError as e:
                raise SlotTaken(str(e)) from e                       # the DB guard caught a race the checks above missed
        return self.get(bid)  # type: ignore[return-value]

    def get(self, booking_id: str) -> Booking | None:
        rows = self.db.query("SELECT * FROM bookings WHERE booking_id = ?", (booking_id,))
        return _booking(rows[0]) if rows else None

    def find_active(self, customer_phone: str, stock_id: str) -> Booking | None:
        rows = self.db.query("SELECT * FROM bookings WHERE customer_phone = ? AND stock_id = ? AND status = 'booked'",
                             (customer_phone, stock_id))
        return _booking(rows[0]) if rows else None

    def reschedule(self, booking_id: str, new_start: datetime, slot_minutes: int, expected_version: int | None = None) -> Booking:
        start, now = _ts(new_start), _ts(datetime.now(timezone.utc))
        with self.db.tx() as c:
            row = c.execute("SELECT * FROM bookings WHERE booking_id = ?", (booking_id,)).fetchone()
            if row is None:
                raise BookingNotFound(booking_id)
            if row["status"] != BookingStatus.BOOKED.value:
                raise NotModifiable(f"booking {booking_id} is {row['status']}")
            if expected_version is not None and row["version"] != expected_version:
                raise StaleBooking(booking_id)
            self._require_available(c, row["stock_id"])
            try:
                c.execute("UPDATE bookings SET slot_start = ?, slot_end = ?, updated_at = ?, version = version + 1 WHERE booking_id = ?",
                          (start, start + slot_minutes * 60, now, booking_id))
            except sqlite3.IntegrityError as e:
                raise SlotTaken(str(e)) from e
        return self.get(booking_id)  # type: ignore[return-value]

    def cancel(self, booking_id: str) -> Booking:
        now = _ts(datetime.now(timezone.utc))
        with self.db.tx() as c:
            row = c.execute("SELECT status FROM bookings WHERE booking_id = ?", (booking_id,)).fetchone()
            if row is None:
                raise BookingNotFound(booking_id)
            if row["status"] == BookingStatus.BOOKED.value:
                c.execute("UPDATE bookings SET status = 'cancelled', updated_at = ?, version = version + 1 WHERE booking_id = ?",
                          (now, booking_id))
        return self.get(booking_id)  # type: ignore[return-value]

    def booked_starts(self, branch_id: str, start: datetime, end: datetime) -> set[datetime]:
        rows = self.db.query("SELECT DISTINCT slot_start FROM bookings WHERE branch_id = ? AND status = 'booked' "
                             "AND slot_start >= ? AND slot_start < ?", (branch_id, _ts(start), _ts(end)))
        return {_dt(r["slot_start"]) for r in rows}
