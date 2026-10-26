r"""The whole system on one machine, the way it runs in production across many: database, one shared model server, N voice workers, router.

  .venv\Scripts\python.exe scripts\run_cluster.py --workers 3

  database   PostgreSQL (pgserver, local; production: a managed PostgreSQL)      bookings, stock, call snapshots, worker registry
  model      python -m dealer_core.modelserver on :8001 (production: vLLM)        one model, batched, for every worker
  workers    python -m dealer_core.server on :8765, :8766, ...                    each up to --calls-per-worker calls
  router     python -m dealer_core.gateway on :8760                                pages ask it which worker to use

A worker that dies is started again (production: Kubernetes does this). Ctrl+C stops everything. The website points at the router with
CONFIG.voiceGateway = "http://localhost:8760" (or ?gateway=http://localhost:8760 in the page address).
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe") if os.name == "nt" else sys.executable


def wait_http(url: str, timeout_s: float = 300.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--calls-per-worker", type=int, default=16)
    ap.add_argument("--first-port", type=int, default=8765)
    ap.add_argument("--model", default="Qwen3-4B-Instruct-2507-int4-ov-nocal")
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL", ""), help="an existing PostgreSQL; default: start one locally")
    ap.add_argument("--quality", default=None)
    a = ap.parse_args()

    procs: dict[str, tuple[list[str], subprocess.Popen]] = {}
    logs = ROOT / "logs" / "cluster"
    logs.mkdir(parents=True, exist_ok=True)

    def launch(name: str, cmd: list[str]) -> None:
        out = open(logs / f"{name}.log", "ab")
        procs[name] = (cmd, subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "DATABASE_URL": db_url}))

    db_url = a.database_url
    if not db_url:
        import pgserver

        db_url = pgserver.get_server(str(ROOT / "data" / "pg_cluster"), cleanup_mode="stop").get_uri()
    print(f"database  {db_url}", flush=True)

    launch("model", [PY, "-m", "dealer_core.modelserver", "--model", a.model, "--port", "8001"])
    if not wait_http("http://127.0.0.1:8001/health"):
        print("the model server did not start: see logs/cluster/model.log", flush=True)
        return 1
    print("model     http://127.0.0.1:8001/v1", flush=True)

    for i in range(a.workers):
        port = a.first_port + i
        cmd = [PY, "-m", "dealer_core.server", "--agent", "--prefix", "--backend", "openvino_server", "--port", str(port),
               "--max-calls", str(a.calls_per_worker), "--state-url", db_url, "--worker-id", f"w{i + 1}"]
        if a.quality:
            cmd += ["--quality", a.quality]
        launch(f"worker{i + 1}", cmd)
    for i in range(a.workers):
        if wait_http(f"http://127.0.0.1:{a.first_port + i}/health"):
            print(f"worker{i + 1}   ws://127.0.0.1:{a.first_port + i}/voice", flush=True)
        else:
            print(f"worker{i + 1} did not start: see logs/cluster/worker{i + 1}.log", flush=True)

    launch("router", [PY, "-m", "dealer_core.gateway", "--state-url", db_url, "--port", "8760"])
    wait_http("http://127.0.0.1:8760/health", 60)
    print(f"router    http://127.0.0.1:8760/assign   ({a.workers} workers x {a.calls_per_worker} calls)", flush=True)
    print("ready", flush=True)

    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            time.sleep(2)
            for name, (cmd, p) in list(procs.items()):
                if p.poll() is not None and not stopping:                  # died: start it again
                    print(f"{name} stopped (exit {p.returncode}); starting it again", flush=True)
                    launch(name, cmd)
    finally:
        for name, (_, p) in procs.items():
            if p.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
                else:
                    p.terminate()
        print("stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
