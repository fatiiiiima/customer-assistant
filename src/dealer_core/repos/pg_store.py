"""PostgreSQL implementation of the inventory and booking repositories: what every voice worker shares in production.

With many voice workers on many machines, bookings must live in one place, and the database itself must make a double booking
impossible, whatever two workers do at the same moment. As in the SQLite version, three partial UNIQUE indexes are the last line of
defence; on top of that, the car's row is locked (SELECT ... FOR UPDATE) while a booking is checked and written, so two callers racing for
the last slot are served one after the other and the second is told the slot has gone.

Driver: pg8000 (BSD, pure Python). Connection string: postgresql://user:password@host:port/database (config `app.database_url` or the
DATABASE_URL environment variable), or postgresql://user:@/database?host=/socket/folder for a server reached through its socket file (the
local PostgreSQL pgserver starts on a Mac or Linux). A small pool keeps connections open for every thread of a worker.
"""
from __future__ import annotations

import json
import queue
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator
from urllib.parse import parse_qs, unquote, urlparse

from ..errors import (BookingNotFound, CarNotFound, CarUnavailable, DuplicateBooking, NotModifiable, SlotTaken,
                      StaleBooking)
from ..models import Booking, BookingStatus, BodyType, Car, CarStatus, NewBooking, SearchFilters

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS cars (
      stock_id TEXT PRIMARY KEY, vin TEXT NOT NULL, make TEXT NOT NULL, model TEXT NOT NULL, year INTEGER NOT NULL,
      km INTEGER NOT NULL, price INTEGER NOT NULL, floor_price INTEGER NOT NULL, status TEXT NOT NULL,
      branch_id TEXT NOT NULL, body_type TEXT NOT NULL, fuel TEXT NOT NULL, transmission TEXT NOT NULL, color TEXT NOT NULL,
      regional_spec TEXT NOT NULL, service_history BOOLEAN NOT NULL, image_urls TEXT NOT NULL,
      CHECK (floor_price <= price AND floor_price > 0))""",
    "CREATE INDEX IF NOT EXISTS ix_cars_search ON cars(status, make, model, price)",
    """CREATE TABLE IF NOT EXISTS bookings (
      booking_id TEXT PRIMARY KEY, stock_id TEXT NOT NULL REFERENCES cars(stock_id), branch_id TEXT NOT NULL,
      customer_name TEXT NOT NULL, customer_phone TEXT NOT NULL, slot_start TIMESTAMPTZ NOT NULL, slot_end TIMESTAMPTZ NOT NULL,
      status TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL, version INTEGER NOT NULL)""",
    # the three double-booking guards; cancelled rows are exempt so a slot can be reused
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_car_slot ON bookings(stock_id, slot_start) WHERE status = 'booked'",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_customer_car ON bookings(customer_phone, stock_id) WHERE status = 'booked'",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_customer_slot ON bookings(customer_phone, slot_start) WHERE status = 'booked'",
    "CREATE INDEX IF NOT EXISTS ix_bookings_branch ON bookings(branch_id, slot_start) WHERE status = 'booked'",
]
_COLS = ("stock_id, vin, make, model, year, km, price, floor_price, status, branch_id, body_type, fuel, transmission, color, "
         "regional_spec, service_history, image_urls")


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware")
    return dt.astimezone(timezone.utc)


def connect_args(url: str) -> dict:
    """pg8000 connection arguments from a postgresql:// address. `?host=/a/folder` means the server's socket file in that folder."""
    u = urlparse(url)
    args = dict(user=unquote(u.username or "postgres"), password=unquote(u.password or "") or None,
                database=(u.path or "/postgres").lstrip("/") or "postgres")
    host = (parse_qs(u.query).get("host") or [""])[0] or u.hostname or "127.0.0.1"
    if host.startswith("/"):
        return {**args, "unix_sock": f"{host.rstrip('/')}/.s.PGSQL.{u.port or 5432}"}
    return {**args, "host": host, "port": u.port or 5432}


_SHARED: dict[str, "PgDb"] = {}
_SHARED_LOCK = threading.Lock()


def shared_db(url: str, size: int = 8) -> "PgDb":
    """The one PgDb of this process for `url`. Bookings, stock and the call state/registry used to open two pools each (8 + 4 connections: about 540 for 45 workers
    against PostgreSQL's default 100); a process now opens at most `size` connections to a database, whoever asks. A later, larger `size` raises the limit."""
    with _SHARED_LOCK:
        db = _SHARED.get(url)
        if db is None:
            db = _SHARED[url] = PgDb(url, size)
        else:
            db._size = max(db._size, size)
        return db


class PgDb:
    """A small pool of pg8000 connections. `tx()` is one transaction; a unique-index violation inside it is re-raised as `_Unique`."""

    def __init__(self, url: str, size: int = 8) -> None:
        import pg8000.native as pg

        self._pg = pg
        self._args = connect_args(url)
        self._pool: queue.Queue = queue.Queue()
        self._size, self._made, self._lock = size, 0, threading.Lock()
        with self.tx() as c:
            c.run("SELECT pg_advisory_xact_lock(424201)")         # workers starting together create the tables one at a time
            for stmt in SCHEMA:
                c.run(stmt)

    def _get(self):
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            with self._lock:
                if self._made < self._size:
                    self._made += 1
                    return self._pg.Connection(**self._args)
            return self._pool.get()

    @contextmanager
    def tx(self) -> Iterator:
        import pg8000.exceptions as pge

        conn, broken = self._get(), False
        try:
            conn.run("BEGIN")
            try:
                yield conn
            except BaseException:
                conn.run("ROLLBACK")
                raise
            conn.run("COMMIT")
        except pge.DatabaseError as e:                      # a statement failed: roll back, keep the connection
            try:
                conn.run("ROLLBACK")
            except Exception:
                broken = True
            info = e.args[0] if e.args and isinstance(e.args[0], dict) else {}
            if info.get("C") == "23505":
                raise _Unique(info.get("n", "")) from e
            raise
        except pge.InterfaceError:                          # the connection itself failed: drop it
            broken = True
            raise
        finally:
            if broken:
                try:
                    conn.close()
                except Exception:
                    pass
                with self._lock:
                    self._made -= 1
            else:
                self._pool.put(conn)

    def rows(self, sql: str, **params) -> list[dict]:
        with self.tx() as c:
            data = c.run(sql, **params)
            names = [col["name"] for col in c.columns] if c.columns else []
        return [dict(zip(names, r)) for r in data or []]


class _Unique(Exception):
    """A unique index refused a write (the constraint name is the argument)."""


def _car(r: dict) -> Car:
    return Car(stock_id=r["stock_id"], vin=r["vin"], make=r["make"], model=r["model"], year=r["year"], km=r["km"], price=r["price"],
               floor_price=r["floor_price"], status=CarStatus(r["status"]), branch_id=r["branch_id"], body_type=BodyType(r["body_type"]),
               fuel=r["fuel"], transmission=r["transmission"], color=r["color"], regional_spec=r["regional_spec"],
               service_history=bool(r["service_history"]), image_urls=tuple(json.loads(r["image_urls"])))


def _booking(r: dict) -> Booking:
    return Booking(booking_id=r["booking_id"], stock_id=r["stock_id"], branch_id=r["branch_id"], customer_name=r["customer_name"],
                   customer_phone=r["customer_phone"], slot_start=r["slot_start"], slot_end=r["slot_end"], status=BookingStatus(r["status"]),
                   created_at=r["created_at"], updated_at=r["updated_at"], version=r["version"])


class PgInventoryRepo:
    def __init__(self, db: PgDb) -> None:
        self.db = db

    def add(self, car: Car) -> None:
        """Insert or refresh a car (every worker loads the same stock at start, so this must be safe to repeat)."""
        with self.db.tx() as c:
            c.run(f"INSERT INTO cars ({_COLS}) VALUES (:a,:b,:c,:d,:e,:f,:g,:h,:i,:j,:k,:l,:m,:n,:o,:p,:q) "
                  "ON CONFLICT (stock_id) DO UPDATE SET make=EXCLUDED.make, model=EXCLUDED.model, year=EXCLUDED.year, km=EXCLUDED.km, "
                  "price=EXCLUDED.price, floor_price=EXCLUDED.floor_price, branch_id=EXCLUDED.branch_id, body_type=EXCLUDED.body_type, "
                  "fuel=EXCLUDED.fuel, transmission=EXCLUDED.transmission, color=EXCLUDED.color, image_urls=EXCLUDED.image_urls",
                  a=car.stock_id, b=car.vin, c=car.make, d=car.model, e=car.year, f=car.km, g=car.price, h=car.floor_price, i=car.status.value,
                  j=car.branch_id, k=car.body_type.value, l=car.fuel, m=car.transmission, n=car.color, o=car.regional_spec,
                  p=car.service_history, q=json.dumps(list(car.image_urls)))

    def mark_status(self, stock_id: str, status: CarStatus) -> None:
        with self.db.tx() as c:
            c.run("UPDATE cars SET status = :s WHERE stock_id = :i", s=status.value, i=stock_id)
            if not c.row_count:
                raise CarNotFound(stock_id)

    @staticmethod
    def _where(f: SearchFilters) -> tuple[str, dict]:
        where, p = [], {}
        if not f.include_unavailable:
            where.append("status = 'available'")
        for col, val, op, key in (("price", f.max_price, "<=", "maxp"), ("price", f.min_price, ">=", "minp"), ("year", f.year_min, ">=", "y"),
                                  ("km", f.max_km, "<=", "km"), ("branch_id", f.branch_id, "=", "br")):
            if val is not None:
                where.append(f"{col} {op} :{key}")
                p[key] = val
        for col, val in (("make", f.make), ("model", f.model), ("color", f.color)):
            if val:
                where.append(f"LOWER({col}) = :{col}")
                p[col] = val.strip().lower()
        if f.body_type:
            where.append("body_type = :body")
            p["body"] = f.body_type.value
        if f.exclude_stock_ids:
            names = []
            for i, sid in enumerate(f.exclude_stock_ids):
                names.append(f":x{i}")
                p[f"x{i}"] = sid
            where.append(f"stock_id NOT IN ({','.join(names)})")
        return (" WHERE " + " AND ".join(where)) if where else "", p

    def search(self, filters: SearchFilters, limit: int = 10) -> list[Car]:
        w, p = self._where(filters)
        return [_car(r) for r in self.db.rows(f"SELECT * FROM cars{w} ORDER BY price, stock_id LIMIT :lim", lim=limit, **p)]

    def count(self, filters: SearchFilters) -> int:
        w, p = self._where(filters)
        return self.db.rows(f"SELECT COUNT(*) AS n FROM cars{w}", **p)[0]["n"]

    def get(self, stock_id: str) -> Car | None:
        rows = self.db.rows("SELECT * FROM cars WHERE stock_id = :i", i=stock_id)
        return _car(rows[0]) if rows else None

    def vocabulary(self) -> dict[str, list[str]]:
        vocab: dict[str, list[str]] = {}
        for r in self.db.rows("SELECT DISTINCT make, model FROM cars WHERE status = 'available' ORDER BY make, model"):
            vocab.setdefault(r["make"].lower(), []).append(r["model"].lower())
        return vocab


class PgBookingRepo:
    def __init__(self, db: PgDb) -> None:
        self.db = db

    @staticmethod
    def _lock_available(c, stock_id: str):
        rows = c.run("SELECT branch_id, status FROM cars WHERE stock_id = :i FOR UPDATE", i=stock_id)   # one booking for this car at a time
        if not rows:
            raise CarNotFound(stock_id)
        branch, status = rows[0]
        if status != CarStatus.AVAILABLE.value:
            raise CarUnavailable(stock_id, status)
        return branch

    def create(self, new: NewBooking, slot_minutes: int) -> Booking:
        start = _utc(new.slot_start)
        now = datetime.now(timezone.utc)
        bid = uuid.uuid4().hex[:12]
        try:
            with self.db.tx() as c:
                branch = self._lock_available(c, new.stock_id)
                dup = c.run("SELECT booking_id FROM bookings WHERE customer_phone = :p AND stock_id = :s AND status = 'booked'",
                            p=new.customer_phone, s=new.stock_id)
                if dup:
                    raise DuplicateBooking(dup[0][0])
                c.run("INSERT INTO bookings VALUES (:id,:s,:b,:n,:p,:st,:en,'booked',:now,:now,1)", id=bid, s=new.stock_id, b=branch,
                      n=new.customer_name, p=new.customer_phone, st=start, en=start + _minutes(slot_minutes), now=now)
        except _Unique as e:
            raise SlotTaken(str(e)) from e                     # the database caught what the checks could not see
        return self.get(bid)  # type: ignore[return-value]

    def get(self, booking_id: str) -> Booking | None:
        rows = self.db.rows("SELECT * FROM bookings WHERE booking_id = :i", i=booking_id)
        return _booking(rows[0]) if rows else None

    def find_active(self, customer_phone: str, stock_id: str) -> Booking | None:
        rows = self.db.rows("SELECT * FROM bookings WHERE customer_phone = :p AND stock_id = :s AND status = 'booked'", p=customer_phone, s=stock_id)
        return _booking(rows[0]) if rows else None

    def reschedule(self, booking_id: str, new_start: datetime, slot_minutes: int, expected_version: int | None = None) -> Booking:
        start, now = _utc(new_start), datetime.now(timezone.utc)
        try:
            with self.db.tx() as c:
                rows = c.run("SELECT stock_id, status, version FROM bookings WHERE booking_id = :i FOR UPDATE", i=booking_id)
                if not rows:
                    raise BookingNotFound(booking_id)
                stock_id, status, version = rows[0]
                if status != BookingStatus.BOOKED.value:
                    raise NotModifiable(f"booking {booking_id} is {status}")
                if expected_version is not None and version != expected_version:
                    raise StaleBooking(booking_id)
                self._lock_available(c, stock_id)
                c.run("UPDATE bookings SET slot_start = :st, slot_end = :en, updated_at = :now, version = version + 1 WHERE booking_id = :i",
                      st=start, en=start + _minutes(slot_minutes), now=now, i=booking_id)
        except _Unique as e:
            raise SlotTaken(str(e)) from e
        return self.get(booking_id)  # type: ignore[return-value]

    def cancel(self, booking_id: str) -> Booking:
        with self.db.tx() as c:
            rows = c.run("SELECT status FROM bookings WHERE booking_id = :i FOR UPDATE", i=booking_id)
            if not rows:
                raise BookingNotFound(booking_id)
            if rows[0][0] == BookingStatus.BOOKED.value:
                c.run("UPDATE bookings SET status = 'cancelled', updated_at = :now, version = version + 1 WHERE booking_id = :i",
                      now=datetime.now(timezone.utc), i=booking_id)
        return self.get(booking_id)  # type: ignore[return-value]

    def booked_starts(self, branch_id: str, start: datetime, end: datetime) -> set[datetime]:
        rows = self.db.rows("SELECT DISTINCT slot_start FROM bookings WHERE branch_id = :b AND status = 'booked' AND slot_start >= :s AND slot_start < :e",
                            b=branch_id, s=_utc(start), e=_utc(end))
        return {r["slot_start"].astimezone(timezone.utc) for r in rows}


def _minutes(n: int):
    from datetime import timedelta

    return timedelta(minutes=n)
