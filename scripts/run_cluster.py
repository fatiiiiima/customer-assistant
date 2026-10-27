r"""The whole system on one machine, the way it runs in production across many: database, the model, N voice workers, router.

  Windows (Intel GPU):  .venv\Scripts\python.exe scripts\run_cluster.py --workers 3
  Mac (Apple chip):     uv run python scripts/run_cluster.py --workers 3          (Ollama running, the model pulled)
  Either, to check:     ... scripts/run_cluster.py --dry-run                      (prints what would start, starts nothing)

  database   PostgreSQL (pgserver, local; production: a managed PostgreSQL)      bookings, stock, call snapshots, worker registry
  model      --backend openvino: python -m dealer_core.modelserver on :8001      one model, batched, for every worker (Intel GPU)
             --backend ollama:   the Ollama app on :11434 (started by you)        Apple chips, or any machine without an Intel GPU
             (default auto: openvino on Intel / AMD processors, ollama otherwise; production: vLLM)
  workers    python -m dealer_core.server on :8765, :8766, ...                    each up to --calls-per-worker calls
  router     python -m dealer_core.gateway on :8760                                pages ask it which worker to use

Needs the `cluster` extra (uv sync ... --extra cluster: pgserver and the PostgreSQL driver). The website is looked for next to this project
(..\bmw-portal\site); --site says where else. A worker that dies is started again (production: Kubernetes does this). Ctrl+C stops
everything. The website points at the router with CONFIG.voiceGateway = "http://localhost:8760" (or ?gateway=http://localhost:8760 in the
page address). Each worker's log: logs/cluster/worker<N>.log.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe") if os.name == "nt" else sys.executable
OLLAMA = "http://127.0.0.1:11434"


def wait_http(url: str, timeout_s: float = 300.0, proc: subprocess.Popen | None = None) -> bool:
    """True once the address answers; False after the timeout, or as soon as `proc` (the process behind it) has exited."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1.5)
    return False


def log_tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def pick_backend(choice: str, machine: str | None = None) -> str:
    """openvino (the shared OpenVINO model server, for an Intel GPU) or ollama (everything else, e.g. Apple chips)."""
    if choice in ("openvino", "ollama"):
        return choice
    m = (machine if machine is not None else platform.machine()).lower()
    return "openvino" if m in ("x86_64", "amd64", "x86", "i386", "i686") else "ollama"


def worker_cmd(i: int, backend: str, db_url: str, site: Path, first_port: int, calls: int, quality: str | None) -> list[str]:
    cmd = [PY, "-m", "dealer_core.server", "--agent", "--prefix", "--backend", "openvino_server" if backend == "openvino" else "ollama",
           "--port", str(first_port + i), "--max-calls", str(calls), "--state-url", db_url, "--worker-id", f"w{i + 1}", "--site", str(site)]
    return cmd + (["--quality", quality] if quality else [])


def worker_env(backend: str, db_url: str) -> dict[str, str]:
    env = {**os.environ, "DATABASE_URL": db_url}
    if backend == "ollama":
        env["VA__DIALOGUE__DECISION_BACKEND"] = "ollama"        # the tool decisions too: the default is the Intel-GPU model
    return env


def ollama_model() -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from dealer_core.config import load_config

    return load_config(overlay=ROOT / "config" / "portal.yaml").llm.backends["ollama"].model


def ollama_problem(model: str) -> str:
    """Why the workers could not use Ollama right now ('' = they can)."""
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=3) as r:
            names = [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return f"Ollama is not answering at {OLLAMA}: open the Ollama app (or run `ollama serve`) and try again."
    if model not in names:
        return f"Ollama does not have {model}: run `ollama pull {model}` and try again (it has: {', '.join(names) or 'nothing'})."
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--calls-per-worker", type=int, default=16)
    ap.add_argument("--first-port", type=int, default=8765)
    ap.add_argument("--backend", choices=["auto", "openvino", "ollama"], default="auto",
                    help="the model: openvino = the shared OpenVINO server (Intel GPU), ollama = the Ollama app; auto picks by processor")
    ap.add_argument("--model", default="Qwen3-4B-Instruct-2507-int4-ov-nocal", help="openvino only: the model folder under models/ov")
    ap.add_argument("--site", type=Path, default=Path(os.environ.get("SITE_DIR") or ROOT.parent / "bmw-portal" / "site"),
                    help="the website's site folder (its data.js holds the cars); default: bmw-portal next to this project")
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL", ""), help="an existing PostgreSQL; default: start one locally")
    ap.add_argument("--quality", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print what would start, start nothing")
    a = ap.parse_args()
    backend = pick_backend(a.backend)

    if not (a.site / "data.js").exists():
        print(f"no website at {a.site} (its data.js): pass --site <the website's site folder>", flush=True)
        if not a.dry_run:
            return 1
    if a.dry_run:
        db = a.database_url or "<a local PostgreSQL, started by pgserver>"
        print(f"backend   {backend}" + (f"  (Ollama at {OLLAMA}, model {ollama_model()})" if backend == "ollama" else ""))
        print(f"database  {db}")
        if backend == "openvino":
            print("model     " + " ".join([PY, "-m", "dealer_core.modelserver", "--model", a.model, "--port", "8001"]))
        for i in range(a.workers):
            extra = "VA__DIALOGUE__DECISION_BACKEND=ollama " if backend == "ollama" else ""
            print(f"worker{i + 1}   {extra}" + " ".join(worker_cmd(i, backend, db, a.site, a.first_port, a.calls_per_worker, a.quality)))
        print("router    " + " ".join([PY, "-m", "dealer_core.gateway", "--state-url", db, "--port", "8760"]))
        return 0

    if backend == "ollama":
        problem = ollama_problem(ollama_model())
        if problem:
            print(problem, flush=True)
            return 1

    procs: dict[str, tuple[list[str], dict, subprocess.Popen]] = {}
    logs = ROOT / "logs" / "cluster"
    logs.mkdir(parents=True, exist_ok=True)

    def launch(name: str, cmd: list[str], env: dict) -> None:
        out = open(logs / f"{name}.log", "ab")
        procs[name] = (cmd, env, subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=env))

    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        raise KeyboardInterrupt                                            # also ends a wait during start-up

    signal.signal(signal.SIGINT, stop)
    if os.name != "nt":
        signal.signal(signal.SIGTERM, stop)                               # `kill`, or the terminal closing
    try:
        db_url = a.database_url
        if not db_url:
            try:
                import pgserver
            except ImportError:
                print("the local PostgreSQL needs the cluster extra: uv sync --python 3.12 --extra voice --extra openvino --extra spike --extra cluster",
                      flush=True)
                return 1
            db_url = pgserver.get_server(str(ROOT / "data" / "pg_cluster"), cleanup_mode="stop").get_uri()
        print(f"database  {db_url}", flush=True)
        base_env = {**os.environ, "DATABASE_URL": db_url}

        if backend == "openvino":
            launch("model", [PY, "-m", "dealer_core.modelserver", "--model", a.model, "--port", "8001"], base_env)
            if not wait_http("http://127.0.0.1:8001/health", proc=procs["model"][2]):
                print("the model server did not start: see logs/cluster/model.log\n" + log_tail(logs / "model.log"), flush=True)
                return 1
            print("model     http://127.0.0.1:8001/v1", flush=True)
        else:
            print(f"model     Ollama at {OLLAMA}", flush=True)

        for i in range(a.workers):
            launch(f"worker{i + 1}", worker_cmd(i, backend, db_url, a.site, a.first_port, a.calls_per_worker, a.quality), worker_env(backend, db_url))
        print(f"starting {a.workers} workers (loading the speech models takes a minute or two)...", flush=True)
        started = 0
        for i in range(a.workers):
            name = f"worker{i + 1}"
            if wait_http(f"http://127.0.0.1:{a.first_port + i}/health", proc=procs[name][2]):
                print(f"{name}   ws://127.0.0.1:{a.first_port + i}/voice", flush=True)
                started += 1
            else:
                print(f"{name} did not start: logs/cluster/{name}.log ends with:\n{log_tail(logs / f'{name}.log')}", flush=True)
        if not started:
            print("no worker started: fix the error above and run this again", flush=True)
            return 1

        launch("router", [PY, "-m", "dealer_core.gateway", "--state-url", db_url, "--port", "8760"], base_env)
        if not wait_http("http://127.0.0.1:8760/health", 60, proc=procs["router"][2]):
            print("the router did not start: logs/cluster/router.log ends with:\n" + log_tail(logs / "router.log"), flush=True)
            return 1
        print(f"router    http://127.0.0.1:8760/assign   ({started} workers x {a.calls_per_worker} calls; live view: http://127.0.0.1:8760/workers)",
              flush=True)
        print("ready", flush=True)

        while not stopping:
            time.sleep(2)
            for name, (cmd, env, p) in list(procs.items()):
                if p.poll() is not None and not stopping:                  # died: start it again
                    print(f"{name} stopped (exit {p.returncode}); starting it again", flush=True)
                    launch(name, cmd, env)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)                      # a second Ctrl+C must not cut the clean-up short
        for name, (_, _, p) in procs.items():
            if p.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
                else:
                    p.terminate()
        print("stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
