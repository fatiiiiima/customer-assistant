r"""Hands each new call to the voice worker with the most free room.

  .venv\Scripts\python.exe -m dealer_core.gateway --state-url postgresql://... --port 8760

The page asks GET /assign before it opens a call and connects straight to the worker it is given (audio never passes through the
router, so the router is never the bottleneck and can run as several copies). Workers report themselves every 2 s; one that stops
reporting is gone from the list within 6 s, so new calls stop going to it. When every worker is full, /assign answers 503 and the page
falls back to the text chat.

  GET /assign    {"url": "ws://.../voice", "worker": "..."}  or 503 {"busy": true}
  GET /workers   every live worker and its load (for dashboards)
  GET /health
"""
from __future__ import annotations

import argparse
import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def create_gateway(registry, allowed_origins: list[str]) -> FastAPI:
    api = FastAPI(title="voice router")
    api.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["GET"], allow_headers=["*"])
    stats = {"assigned": 0, "busy": 0, "started": time.time()}

    @api.get("/assign")
    async def assign():
        w = registry.pick()
        if w is None:
            stats["busy"] += 1
            return JSONResponse({"busy": True}, status_code=503)
        stats["assigned"] += 1
        return {"url": w["url"], "worker": w["id"]}

    @api.get("/workers")
    async def workers():
        ws = registry.workers()
        return {"workers": sorted(ws, key=lambda w: w["id"]), "calls": sum(w.get("active", 0) for w in ws),
                "capacity": sum(w.get("max", 0) for w in ws)}

    @api.get("/health")
    async def health():
        return {"ok": True, **stats, "workers": len(registry.workers())}

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
