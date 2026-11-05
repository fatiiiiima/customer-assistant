r"""The whole system on one machine, the way it runs in production across many: database, the model, N voice workers, router.

  Windows (Intel GPU):  .venv\Scripts\python.exe scripts\run_cluster.py --workers 3
  Mac (Apple chip):     uv run python scripts/run_cluster.py --workers 3          (Ollama running, the model pulled)
  Either, to check:     ... scripts/run_cluster.py --dry-run                      (prints what would start, starts nothing)

  database   PostgreSQL (pgserver, local; production: a managed PostgreSQL)      bookings, stock, call snapshots, worker registry
  model      --backend bedrock:  Nemotron on Amazon Bedrock (dev only: the callers' words go to AWS; needs the key in .env) and the live settings of the owner's calls:
                                 hands-free, --quality full; the rest with --set VA__VOICE__MODEL_ONLY=true ... (repeat it), the recogniser with --set VA__VOICE__STT_URL=...
             --backend openvino: python -m dealer_core.modelserver on :8001      one model, batched, for every worker (Intel GPU)
             --backend ollama:   the Ollama app on :11434 (started by you)        Apple chips, or any machine without an Intel GPU
             (default auto: openvino on Intel / AMD processors, ollama otherwise; production: vLLM)
  workers    python -m dealer_core.server on :8765, :8766, ...                    each up to --calls-per-worker calls
  router     python -m dealer_core.gateway on :8760                                pages ask it which worker to use
  redis      the workers' load reports and the call snapshots (kept a day, on disk in data/redis), not in PostgreSQL, which holds only the stock and bookings (--no-redis: back in
             PostgreSQL; --redis-url: a Redis you already have)
  recognizer (with --stt-server) python -m dealer_core.sttserver on :8002          ONE speech recognizer for every worker (they use VA__VOICE__STT_URL), like a GPU server in production

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
LOCAL_SERVER = "http://127.0.0.1:8003/v1"          # llama-server / mlx_lm.server (config backend local_server)


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
    """openvino (the shared OpenVINO model server, for an Intel GPU), ollama (everything else, e.g. Apple chips), local (an OpenAI-compatible server you started) or bedrock
    (Nemotron on Amazon Bedrock, the way the owner's live calls run: ARCH_REVIEW section 99)."""
    if choice in ("openvino", "ollama", "local", "bedrock"):
        return choice
    m = (machine if machine is not None else platform.machine()).lower()
    return "openvino" if m in ("x86_64", "amd64", "x86", "i386", "i686") else "ollama"


def worker_cmd(i: int, backend: str, db_url: str, site: Path, first_port: int, calls: int, quality: str | None, state_url: str = "") -> list[str]:
    """`state_url` is where the workers report their load and keep call snapshots (a redis:// address); empty: the database, as before. Bookings and stock stay in `db_url`."""
    tail = ["--port", str(first_port + i), "--max-calls", str(calls), "--state-url", state_url or db_url, "--worker-id", f"w{i + 1}", "--site", str(site)]
    if backend == "bedrock":                                              # as the owner's live calls: the model writes the reply, the microphone stays open
        return [PY, "-m", "dealer_core.server", "--agent", "--quality", quality or "full", "--hands-free", "--backend", "bedrock_nemotron", *tail]
    cmd = [PY, "-m", "dealer_core.server", "--agent", "--prefix", "--backend", {"openvino": "openvino_server", "local": "local_server"}.get(backend, "ollama"), *tail]
    return cmd + (["--quality", quality] if quality else [])


STT_PORT = 8002
REDIS_PORT = 6390


def redis_server_cmd(port: int = REDIS_PORT, data_dir: Path | None = None) -> list[str] | None:
    """The command that starts a local Redis for the workers' load reports and the call snapshots (None if redis-server is not installed). It keeps an append-only file in
    data/redis (written once a second), so the conversations it holds, kept a day, are still there after Redis or the machine restarts."""
    import shutil

    exe = shutil.which("redis-server")
    folder = data_dir or ROOT / "data" / "redis"
    return [exe, "--port", str(port), "--bind", "127.0.0.1", "--dir", str(folder), "--save", "", "--appendonly", "yes", "--appendfsync", "everysec", "--daemonize", "no"] if exe else None


def parse_sets(items: list[str] | None) -> dict[str, str]:
    """--set KEY=VALUE (repeatable): environment for every worker (the VA__SECTION__KEY settings of config/default.yaml)."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--set wants KEY=VALUE, not {item!r}")
        out[key.strip()] = value
    return out


def worker_env(backend: str, db_url: str, stt_url: str = "", local_model: str = "", extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {**os.environ, "DATABASE_URL": db_url}
    if backend == "bedrock":
        env.update({"VOICE_AGENT_ALLOW_CLOUD": "1", "VA__DIALOGUE__DECISION_BACKEND": "bedrock_nemotron", "VA__LLM__FALLBACKS": "[]"})
    if stt_url:
        env["VA__VOICE__STT_URL"] = stt_url                          # the workers then run no recognizer of their own (voice/stt.py RemoteWhisper)
    if backend == "ollama":
        env["VA__DIALOGUE__DECISION_BACKEND"] = "ollama"        # the tool decisions too: the default is the Intel-GPU model
    if backend == "local":
        env["VA__DIALOGUE__DECISION_BACKEND"] = "local_server"
        if local_model:
            env["VA__LLM__BACKENDS__LOCAL_SERVER__MODEL"] = local_model     # the name the server lists
    env.update(extra or {})
    return env


def local_server_model(url: str = LOCAL_SERVER) -> tuple[str, str]:
    """(the model name the local server lists, "") or ("", why the workers cannot use it)."""
    try:
        with urllib.request.urlopen(f"{url}/models", timeout=3) as r:
            names = [m.get("id", "") for m in json.loads(r.read()).get("data", [])]
    except Exception:
        return "", (f"No model server is answering at {url}. Start one (docs/ARCH_REVIEW.md section 7b), for example on a Mac:\n"
                    "  mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8003\nor with llama.cpp:\n"
                    "  llama-server -m <qwen3-4b-instruct-2507 Q4_K_M .gguf> --port 8003 --parallel 4 -c 8192")
    return (names[0], "") if names else ("", f"The server at {url} lists no model.")


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
    ap.add_argument("--backend", choices=["auto", "openvino", "ollama", "local", "bedrock"], default="auto",
                    help="the model: openvino = the shared OpenVINO server (Intel GPU), ollama = the Ollama app, bedrock = Nemotron on Amazon Bedrock (dev only), "
                         "local = a server you started; auto picks by processor")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="an environment setting for every worker, e.g. --set VA__VOICE__HOLD_REPLY=true (repeatable)")
    ap.add_argument("--model", default="Qwen3-4B-Instruct-2507-int4-ov-nocal", help="openvino only: the model folder under models/ov")
    ap.add_argument("--site", type=Path, default=Path(os.environ.get("SITE_DIR") or ROOT.parent / "bmw-portal" / "site"),
                    help="the website's site folder (its data.js holds the cars); default: bmw-portal next to this project")
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL", ""), help="an existing PostgreSQL; default: start one locally")
    ap.add_argument("--redis-url", default="", help="the workers' load reports (the router's list) and the call snapshots live in Redis, not in the database that holds the stock and bookings. "
                                                    "Default: start a local redis-server (it keeps its data in data/redis); with this, one you already have (redis://host:6379/0)")
    ap.add_argument("--no-redis", action="store_true", help="keep them in the database instead, as before 5 Nov")
    ap.add_argument("--quality", default=None)
    ap.add_argument("--stt-server", action="store_true", help="run one shared speech recognizer (dealer_core.sttserver) instead of one Whisper per worker")
    ap.add_argument("--dry-run", action="store_true", help="print what would start, start nothing")
    a = ap.parse_args()
    backend = pick_backend(a.backend)
    extra = parse_sets(a.set)

    if not (a.site / "data.js").exists():
        print(f"no website at {a.site} (its data.js): pass --site <the website's site folder>", flush=True)
        if not a.dry_run:
            return 1
    if a.dry_run:
        db = a.database_url or "<a local PostgreSQL, started by pgserver>"
        print(f"backend   {backend}" + (f"  (Ollama at {OLLAMA}, model {ollama_model()})" if backend == "ollama" else f"  (an OpenAI-compatible server you start, at {LOCAL_SERVER})" if backend == "local" else ""))
        print(f"database  {db}")
        state = "" if a.no_redis else a.redis_url or f"redis://127.0.0.1:{REDIS_PORT}/0"
        if not a.no_redis and not a.redis_url:
            print("redis     " + (" ".join(redis_server_cmd() or []) or "redis-server is not installed (brew install redis)"))
        if state:
            print(f"state     {state}   (the workers' load reports and call snapshots; bookings and stock stay in the database)")
        if backend == "openvino":
            print("model     " + " ".join([PY, "-m", "dealer_core.modelserver", "--model", a.model, "--port", "8001"]))
        if a.stt_server:
            print("recogniz  " + " ".join([PY, "-m", "dealer_core.sttserver", "--port", str(STT_PORT)]) + f"   (workers: VA__VOICE__STT_URL=http://127.0.0.1:{STT_PORT}/v1)")
        for i in range(a.workers):
            shown = {"ollama": "VA__DIALOGUE__DECISION_BACKEND=ollama ", "local": "VA__DIALOGUE__DECISION_BACKEND=local_server ",
                     "bedrock": "VOICE_AGENT_ALLOW_CLOUD=1 VA__DIALOGUE__DECISION_BACKEND=bedrock_nemotron "}.get(backend, "") + "".join(f"{k}={v} " for k, v in extra.items())
            print(f"worker{i + 1}   {shown}" + " ".join(worker_cmd(i, backend, db, a.site, a.first_port, a.calls_per_worker, a.quality, state)))
        print("router    " + " ".join([PY, "-m", "dealer_core.gateway", "--state-url", state or db, "--port", "8760"]))
        return 0

    if backend == "ollama":
        problem = ollama_problem(ollama_model())
        if problem:
            print(problem, flush=True)
            return 1
    local_model = ""
    if backend == "local":
        local_model, problem = local_server_model()
        if problem:
            print(problem, flush=True)
            return 1
        print(f"model     {local_model} at {LOCAL_SERVER}", flush=True)

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
        state_url = "" if a.no_redis else a.redis_url
        if not a.no_redis and not state_url:
            cmd = redis_server_cmd()
            if cmd is None:
                print("redis-server is not installed: brew install redis (or pass --redis-url for one you have, or --no-redis to keep the state in the database)", flush=True)
                return 1
            (ROOT / "data" / "redis").mkdir(parents=True, exist_ok=True)
            launch("redis", cmd, base_env)
            time.sleep(1)
            if procs["redis"][2].poll() is not None:
                print("redis did not start: see logs/cluster/redis.log\n" + log_tail(logs / "redis.log"), flush=True)
                return 1
            state_url = f"redis://127.0.0.1:{REDIS_PORT}/0"
        if state_url:
            print(f"state     {state_url}   (load reports and call snapshots; bookings and stock stay in the database)", flush=True)

        if backend == "openvino":
            launch("model", [PY, "-m", "dealer_core.modelserver", "--model", a.model, "--port", "8001"], base_env)
            if not wait_http("http://127.0.0.1:8001/health", proc=procs["model"][2]):
                print("the model server did not start: see logs/cluster/model.log\n" + log_tail(logs / "model.log"), flush=True)
                return 1
            print("model     http://127.0.0.1:8001/v1", flush=True)
        elif backend == "ollama":
            print(f"model     Ollama at {OLLAMA}", flush=True)

        stt_url = ""
        if a.stt_server:
            launch("stt", [PY, "-m", "dealer_core.sttserver", "--port", str(STT_PORT)], base_env)
            if not wait_http(f"http://127.0.0.1:{STT_PORT}/health", proc=procs["stt"][2]):
                print("the speech recognizer did not start: see logs/cluster/stt.log\n" + log_tail(logs / "stt.log"), flush=True)
                return 1
            stt_url = f"http://127.0.0.1:{STT_PORT}/v1"
            print(f"recogniz  {stt_url}", flush=True)
        for i in range(a.workers):
            launch(f"worker{i + 1}", worker_cmd(i, backend, db_url, a.site, a.first_port, a.calls_per_worker, a.quality, state_url), worker_env(backend, db_url, stt_url, local_model, extra))
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

        launch("router", [PY, "-m", "dealer_core.gateway", "--state-url", state_url or db_url, "--port", "8760"], base_env)
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
