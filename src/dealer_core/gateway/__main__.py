r"""Hands each new call to the voice worker with the most free room.

  .venv\Scripts\python.exe -m dealer_core.gateway --state-url postgresql://... --port 8760

The page asks GET /assign before it opens a call and connects straight to the worker it is given (audio never passes through the
router, so the router is never the bottleneck and can run as several copies). Workers report themselves every 2 s (and soon after a call
starts or ends); one that stops reporting is gone from the list within 6 s, so new calls stop going to it. When every worker is full,
/assign answers 503 and the page falls back to the text chat.

Built for bursts (hundreds of callers arriving within seconds):
  * /assign never touches the database: the worker list is read in the background every 0.5 s and kept in memory.
  * Calls this router handed out are counted against a worker until its own report can include them, so a burst does not all go to the
    one worker that looked emptiest 2 s ago (and then hear "busy").
  * Power of two choices: two workers with room are drawn at random and the emptier one gets the call, so several router copies spread
    the same burst instead of all picking the same "least busy" worker.

  GET /assign    {"url": "ws://.../voice", "worker": "..."}  or 503 {"busy": true}
  GET /workers   every live worker and its load (for dashboards)
  GET /health
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import threading
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


class Assigner:
    """The router's in-memory view of the workers, and the calls it handed out that their reports cannot show yet."""

    def __init__(self, registry, hold_s: float = 3.0, rng: random.Random | None = None, clock=time.monotonic) -> None:
        self.registry, self.hold_s = registry, hold_s           # hold: one report interval (2 s) + the page's time to connect
        self.rng, self.clock = rng or random.Random(), clock
        self.workers: list[dict] = []
        self.refreshed = 0.0
        self._held: dict[str, list[float]] = {}                 # worker id -> when each call was handed to it
        self._lock = threading.Lock()

    def refresh(self) -> None:
        """Read every worker's report (blocking: call it from a thread)."""
        ws = self.registry.workers()
        with self._lock:
            self.workers, self.refreshed = ws, self.clock()

    def _reserved(self, wid: str, now: float) -> int:
        held = [t for t in self._held.get(wid, []) if now - t < self.hold_s]
        if held:
            self._held[wid] = held
        else:
            self._held.pop(wid, None)
        return len(held)

    def pick(self) -> dict | None:
        """A live worker with room, or None if all are full. Counts the call against that worker at once."""
        with self._lock:
            now = self.clock()
            room = [(w, w.get("max", 0) - w.get("active", 0) - self._reserved(w["id"], now)) for w in self.workers]
            room = [(w, free) for w, free in room if free > 0]
            if not room:
                return None
            a, b = self.rng.sample(room, 2) if len(room) >= 2 else (room[0], room[0])
            w = a[0] if a[1] >= b[1] else b[0]
            self._held.setdefault(w["id"], []).append(now)
            return w


def create_gateway(registry, allowed_origins: list[str], refresh_s: float = 0.5, assigner: Assigner | None = None) -> FastAPI:
    api = FastAPI(title="voice router")
    api.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["GET"], allow_headers=["*"])
    stats = {"assigned": 0, "busy": 0, "started": time.time()}
    view = assigner or Assigner(registry)
    api.state.assigner = view

    @api.on_event("startup")
    async def _watch() -> None:
        await asyncio.to_thread(view.refresh)

        async def loop() -> None:
            while True:
                await asyncio.sleep(refresh_s)
                try:
                    await asyncio.to_thread(view.refresh)
                except Exception:
                    pass                                        # keep the last view; workers that died still expire from it on the next good read

        asyncio.get_running_loop().create_task(loop())

    @api.get("/assign")
    async def assign():
        w = view.pick()
        if w is None:
            stats["busy"] += 1
            return JSONResponse({"busy": True}, status_code=503)
        stats["assigned"] += 1
        return {"url": w["url"], "worker": w["id"]}

    @api.get("/workers")
    async def workers():
        ws = view.workers
        return {"workers": sorted(ws, key=lambda w: w["id"]), "calls": sum(w.get("active", 0) for w in ws),
                "capacity": sum(w.get("max", 0) for w in ws)}

    @api.get("/health")
    async def health():
        return {"ok": True, **stats, "workers": len(view.workers), "view_age_s": round(time.monotonic() - view.refreshed, 2)}

    return api


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-url", default=os.environ.get("STATE_URL") or os.environ.get("DATABASE_URL", ""))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--origin", action="append")
    a = ap.parse_args()
    if not a.state_url:
        raise SystemExit("the router needs the shared state the workers report to: --state-url postgresql://... or redis://...")
    import uvicorn

    from ..shared.state import WorkerRegistry, open_store

    reg = WorkerRegistry(open_store(a.state_url))
    origins = a.origin or ["http://localhost:8080", "http://127.0.0.1:8080"]
    print(f"Router ready on http://{a.host}:{a.port}/assign", flush=True)
    uvicorn.run(create_gateway(reg, origins), host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
