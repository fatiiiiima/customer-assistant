"""A tiny key-value store with expiry, and the two things the voice workers keep in it.

  StateStore     get / set (with a time-to-live) / delete / keys by prefix.  RedisStore (redis-py, MIT) or MemoryStore.
  CallStore      the conversation state of each call, saved after every turn, so another worker can carry on if this one dies.
  WorkerRegistry every worker's address and load, refreshed every few seconds; a worker that stops refreshing drops out by itself.

`open_store(url)`: "redis://host:6379/0" -> Redis; "" -> in-process memory (one machine only).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from datetime import date, datetime


class StateStore:
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl_s: float | None = None) -> None: ...
    def delete(self, key: str) -> None: ...
    def keys(self, prefix: str) -> list[str]: ...

    def items(self, prefix: str) -> dict[str, str]:
        """Every live key with this prefix and its value. The default asks for each value in turn (N + 1 reads); stores that can do it in one read override it."""
        out = {}
        for k in self.keys(prefix):
            v = self.get(k)
            if v is not None:
                out[k] = v
        return out


class MemoryStore(StateStore):
    def __init__(self) -> None:
        self._d: dict[str, tuple[str, float | None]] = {}
        self._lock = threading.Lock()

    def _alive(self, key: str) -> bool:
        v = self._d.get(key)
        if v is None:
            return False
        if v[1] is not None and v[1] < time.monotonic():
            self._d.pop(key, None)
            return False
        return True

    def get(self, key):
        with self._lock:
            return self._d[key][0] if self._alive(key) else None

    def set(self, key, value, ttl_s=None):
        with self._lock:
            self._d[key] = (value, time.monotonic() + ttl_s if ttl_s else None)

    def delete(self, key):
        with self._lock:
            self._d.pop(key, None)

    def keys(self, prefix):
        with self._lock:
            return [k for k in list(self._d) if k.startswith(prefix) and self._alive(k)]


class RedisStore(StateStore):
    def __init__(self, url: str = "", client=None) -> None:
        if client is None:
            import redis

            client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2.0, socket_connect_timeout=2.0)
        self.r = client

    def get(self, key):
        return self.r.get(key)

    def set(self, key, value, ttl_s=None):
        self.r.set(key, value, px=int(ttl_s * 1000) if ttl_s else None)

    def delete(self, key):
        self.r.delete(key)

    def keys(self, prefix):
        return [k for k in self.r.scan_iter(match=prefix + "*", count=500)]

    def items(self, prefix):
        ks = self.keys(prefix)
        return {k: v for k, v in zip(ks, self.r.mget(ks) if ks else []) if v is not None}


class PgStore(StateStore):
    """The same store in PostgreSQL (an UNLOGGED table: fast, and nothing in it needs to survive a database crash). With PostgreSQL already
    holding the bookings, production needs no separate Redis: 500 calls saving a snapshot every ~15 s is ~35 small writes a second."""

    TIDY_EVERY_S = 30.0

    def __init__(self, url: str, size: int = 4) -> None:
        from ..repos.pg_store import shared_db

        self.db = shared_db(url, size)                     # the process's one pool for this database (bookings share it)
        self._tidied = 0.0
        with self.db.tx() as c:
            c.run("SELECT pg_advisory_xact_lock(424202)")
            c.run("CREATE UNLOGGED TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL, expires_at TIMESTAMPTZ)")

    def get(self, key):
        rows = self.db.rows("SELECT value FROM kv WHERE key = :k AND (expires_at IS NULL OR expires_at > now())", k=key)
        return rows[0]["value"] if rows else None

    def set(self, key, value, ttl_s=None):
        with self.db.tx() as c:
            c.run("INSERT INTO kv VALUES (:k, :v, CASE WHEN CAST(:t AS DOUBLE PRECISION) IS NULL THEN NULL "
                  "ELSE now() + make_interval(secs => CAST(:t AS DOUBLE PRECISION)) END) "
                  "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, expires_at = EXCLUDED.expires_at", k=key, v=value, t=ttl_s)

    def delete(self, key):
        with self.db.tx() as c:
            c.run("DELETE FROM kv WHERE key = :k", k=key)

    def _tidy(self) -> None:
        """Delete long-expired rows, at most every TIDY_EVERY_S: the router reads the registry every 0.5 s, and a DELETE per read was a write per read."""
        now = time.monotonic()
        if now - self._tidied < self.TIDY_EVERY_S:
            return
        self._tidied = now
        with self.db.tx() as c:
            c.run("DELETE FROM kv WHERE expires_at IS NOT NULL AND expires_at < now() - interval '1 minute'")

    def keys(self, prefix):
        self._tidy()
        return [r["key"] for r in self.db.rows("SELECT key FROM kv WHERE key LIKE :p AND (expires_at IS NULL OR expires_at > now())", p=prefix + "%")]

    def items(self, prefix):
        """Keys and values in ONE read (the registry used to read a key list and then each worker's row in turn: 1 + N queries every 0.5 s per router)."""
        self._tidy()
        return {r["key"]: r["value"] for r in self.db.rows(
            "SELECT key, value FROM kv WHERE key LIKE :p AND (expires_at IS NULL OR expires_at > now())", p=prefix + "%")}


def open_store(url: str = "", pool_size: int = 4) -> StateStore:
    """redis://... -> Redis; postgresql://... -> PostgreSQL; empty -> in-process memory (one process only)."""
    if url.startswith(("redis://", "rediss://")):
        return RedisStore(url)
    if url.startswith(("postgresql://", "postgres://")):
        return PgStore(url, pool_size)
    return MemoryStore()


# ------------------------------------------------------------------------------------------------ call snapshots
def _plain(v):
    if isinstance(v, datetime):
        return {"__dt__": v.isoformat()}
    if isinstance(v, date):
        return {"__d__": v.isoformat()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def _back(v):
    if isinstance(v, dict):
        if "__dt__" in v:
            return datetime.fromisoformat(v["__dt__"])
        if "__d__" in v:
            return date.fromisoformat(v["__d__"])
        return {k: _back(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_back(x) for x in v]
    return v


def snapshot(session) -> dict:
    """What another worker needs to carry on this call. Cars are kept by id (the stock is in the shared database)."""
    from ..dialogue.session import Session  # noqa: F401  (type only)

    s = session
    return _plain({
        "state": s.state.value, "car": s.car, "shown": [c.stock_id for c in s.shown], "filters": dict(s.filters), "narrow_asks": s.narrow_asks,
        "offered_slots": list(s.offered_slots), "slot_day": s.slot_day, "draft_slot": s.draft_slot, "draft_name": s.draft_name,
        "draft_phone": s.draft_phone, "booking_id": s.booking.booking_id if s.booking else None, "read_back": s.read_back,
        "pending": {"kind": s.pending.kind, "value": s.pending.value} if s.pending else None, "ended": s.ended, "history": [list(h) for h in s.history],
        "known_bookings": [asdict(k) for k in s.known_bookings], "pending_cancel": asdict(s.pending_cancel) if s.pending_cancel else None,
        "replacing": asdict(s.replacing) if s.replacing else None, "cancelled": [asdict(k) for k in s.cancelled], "described": s.described,
        "makes_said": list(s.makes_said), "last_variant": dict(s.last_variant), "day_asks": s.day_asks,
        "no_invite_until": s.no_invite_until, "said_about": {k: list(v) for k, v in s.said_about.items()},
        "haggles": {k: {**v, "offered": list(v["offered"])} for k, v in s.haggles.items()},
        "forgotten": s.forgotten,
    })


def restore(data: dict, session, service) -> None:
    """Put a snapshot back into a fresh Session (the cars are looked up again in the shared stock)."""
    from ..dialogue.session import KnownBooking, Pending
    from ..dialogue.states import State

    d = _back(data)
    s = session
    s.state = State(d["state"])
    s.car = d["car"]
    s.shown = [c for c in (service.get_car(i) for i in d["shown"]) if c is not None]
    s.filters, s.narrow_asks = d["filters"], d["narrow_asks"]
    s.offered_slots, s.slot_day = d["offered_slots"], d["slot_day"]
    s.draft_slot, s.draft_name, s.draft_phone, s.read_back = d["draft_slot"], d["draft_name"], d["draft_phone"], d["read_back"]
    if d.get("booking_id"):
        b = service.bookings.get(d["booking_id"])
        s.booking = service._view(b) if b is not None and hasattr(service, "_view") else None
    pv = d.get("pending")
    s.pending = Pending(pv["kind"], pv["value"]) if pv else None
    s.ended = d["ended"]
    s.history = [tuple(h) for h in d["history"]]
    s.known_bookings = [KnownBooking(**k) for k in d["known_bookings"]]
    s.pending_cancel = KnownBooking(**d["pending_cancel"]) if d.get("pending_cancel") else None
    s.replacing = KnownBooking(**d["replacing"]) if d.get("replacing") else None
    s.cancelled = [KnownBooking(**k) for k in d["cancelled"]]
    s.described, s.makes_said, s.last_variant = d["described"], d["makes_said"], d["last_variant"]
    s.day_asks = d.get("day_asks", 0)
    s.no_invite_until, s.said_about = d.get("no_invite_until", 0), {k: list(v) for k, v in (d.get("said_about") or {}).items()}
    s.haggles = {k: {**v, "offered": list(v["offered"])} for k, v in (d.get("haggles") or {}).items()}
    s.forgotten = d.get("forgotten", 0)


class CallStore:
    """Conversation state per call id. Kept for `ttl_s` after the last turn, so a caller who drops and reconnects carries on."""

    def __init__(self, store: StateStore, ttl_s: float = 900.0) -> None:
        self.store, self.ttl = store, ttl_s

    def save(self, call_id: str, session) -> None:
        try:
            self.store.set(f"call:{call_id}", json.dumps(snapshot(session)), self.ttl)
        except Exception:
            pass                                   # never let the shared store break a live call

    def save_json(self, call_id: str, data: str) -> None:
        try:
            self.store.set(f"call:{call_id}", data, self.ttl)
        except Exception:
            pass

    def exists(self, call_id: str) -> bool:
        try:
            return bool(self.store.get(f"call:{call_id}"))
        except Exception:
            return False

    def load(self, call_id: str, session, service) -> bool:
        try:
            raw = self.store.get(f"call:{call_id}")
            if not raw:
                return False
            restore(json.loads(raw), session, service)
            return True
        except Exception:
            return False

    def drop(self, call_id: str) -> None:
        try:
            self.store.delete(f"call:{call_id}")
        except Exception:
            pass


class AsyncCallSaver:
    """Saves one call's snapshot after every turn WITHOUT blocking the worker's event loop (which carries every call's audio).

    The snapshot is taken on the loop at once (cheap, and consistent with the turn that just ended); the write to Redis / PostgreSQL runs in
    a thread. Writes for one call never overlap and never go backwards: while one is in flight only the newest snapshot waits, older ones are
    dropped. `flush()` at the end of the call waits for the last one, so a reconnect finds the latest state."""

    def __init__(self, calls: CallStore, call_id: str) -> None:
        import asyncio

        self.calls, self.call_id = calls, call_id
        self._pending: str | None = None
        self._task: asyncio.Task | None = None
        self.writes = 0

    def save(self, session) -> None:
        import asyncio

        try:
            self._pending = json.dumps(snapshot(session))
        except Exception:
            return
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._drain())

    async def _drain(self) -> None:
        import asyncio

        while self._pending is not None:
            data, self._pending = self._pending, None
            await asyncio.to_thread(self.calls.save_json, self.call_id, data)
            self.writes += 1

    async def flush(self) -> None:
        if self._task is not None:
            await self._task


# ------------------------------------------------------------------------------------------------ worker registry
class WorkerRegistry:
    """Each worker writes `worker:<id>` = {url, active, max, quality} every few seconds with a short expiry; the router reads them all."""

    def __init__(self, store: StateStore, ttl_s: float = 6.0) -> None:
        self.store, self.ttl = store, ttl_s

    def beat(self, worker_id: str, url: str, active: int, max_calls: int, extra: dict | None = None) -> None:
        try:
            self.store.set(f"worker:{worker_id}", json.dumps({"id": worker_id, "url": url, "active": active, "max": max_calls, **(extra or {}),
                                                              "at": time.time()}), self.ttl)
        except Exception:
            pass

    def leave(self, worker_id: str) -> None:
        self.store.delete(f"worker:{worker_id}")

    def workers(self) -> list[dict]:
        out = []
        for raw in self.store.items("worker:").values():
            if raw:
                try:
                    out.append(json.loads(raw))
                except ValueError:
                    pass
        return out

    def pick(self) -> dict | None:
        """The live worker with the most free room (none if all are full). A worker that is draining (shutting down) is never picked."""
        free = [w for w in self.workers() if not w.get("draining") and w.get("active", 0) < w.get("max", 0)]
        return max(free, key=lambda w: (w["max"] - w["active"], -w["active"])) if free else None
