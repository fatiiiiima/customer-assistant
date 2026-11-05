"""scripts/run_cluster.py: the model by processor (OpenVINO server on Intel / AMD, Ollama elsewhere), what each worker is started with."""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_cluster", ROOT / "scripts" / "run_cluster.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


def test_the_model_backend_follows_the_processor_unless_chosen():
    assert rc.pick_backend("auto", "AMD64") == "openvino" and rc.pick_backend("auto", "x86_64") == "openvino"
    assert rc.pick_backend("auto", "arm64") == "ollama"                           # Apple chips: no Intel GPU
    assert rc.pick_backend("ollama", "AMD64") == "ollama" and rc.pick_backend("openvino", "arm64") == "openvino"


def test_an_ollama_worker_uses_ollama_for_replies_and_decisions_and_finds_the_website(monkeypatch):
    monkeypatch.delenv("VA__DIALOGUE__DECISION_BACKEND", raising=False)
    cmd = rc.worker_cmd(1, "ollama", "postgresql://db", Path("/home/f/dev/bmw-portal/site"), 8765, 16, None)
    assert cmd[cmd.index("--backend") + 1] == "ollama" and cmd[cmd.index("--port") + 1] == "8766"
    assert cmd[cmd.index("--site") + 1] == "/home/f/dev/bmw-portal/site" and cmd[cmd.index("--state-url") + 1] == "postgresql://db"
    env = rc.worker_env("ollama", "postgresql://db")
    assert env["VA__DIALOGUE__DECISION_BACKEND"] == "ollama" and env["DATABASE_URL"] == "postgresql://db"
    win = rc.worker_cmd(0, "openvino", "postgresql://db", Path("C:/dev/bmw-portal/site"), 8765, 16, "prefix")
    assert win[win.index("--backend") + 1] == "openvino_server" and win[-2:] == ["--quality", "prefix"]
    assert "VA__DIALOGUE__DECISION_BACKEND" not in rc.worker_env("openvino", "x")                    # Windows keeps its own decision model


def test_a_dry_run_prints_the_plan_and_starts_nothing(tmp_path):
    (tmp_path / "data.js").write_text("// cars")
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_cluster.py"), "--dry-run", "--backend", "ollama", "--workers", "3",
                          "--site", str(tmp_path)], capture_output=True, text=True, timeout=60).stdout
    assert "backend   ollama" in out and "qwen3:4b" in out and "model     " not in out              # no model server is started
    assert out.count("dealer_core.server") == 3 and "VA__DIALOGUE__DECISION_BACKEND=ollama" in out and "dealer_core.gateway" in out
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_cluster.py"), "--dry-run", "--backend", "openvino", "--site", str(tmp_path)],
                         capture_output=True, text=True, timeout=60).stdout
    assert "dealer_core.modelserver" in out and "openvino_server" in out


def test_waiting_for_a_worker_stops_as_soon_as_it_has_died():
    dead = subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"])
    dead.wait()
    t0 = __import__("time").time()
    assert rc.wait_http("http://127.0.0.1:9/health", timeout_s=30, proc=dead) is False
    assert __import__("time").time() - t0 < 2                                       # not the 30 s timeout


def test_the_database_address_of_a_local_postgresql_on_a_mac_is_its_socket_file():
    from dealer_core.repos.pg_store import connect_args

    mac = connect_args("postgresql://postgres:@/postgres?host=/Users/f/dev/voice-agent/data/pg_cluster")   # what pgserver gives on a Mac
    assert mac["unix_sock"] == "/Users/f/dev/voice-agent/data/pg_cluster/.s.PGSQL.5432" and "host" not in mac
    assert mac["user"] == "postgres" and mac["password"] is None and mac["database"] == "postgres"
    win = connect_args("postgresql://postgres:@127.0.0.1:5433/postgres")                                   # what pgserver gives on Windows
    assert win["host"] == "127.0.0.1" and win["port"] == 5433 and "unix_sock" not in win
    assert connect_args("postgresql://app:p%40ss@db.internal/voice") == {"user": "app", "password": "p@ss", "database": "voice",
                                                                        "host": "db.internal", "port": 5432}


def test_a_local_model_server_is_chosen_with_backend_local_and_used_for_replies_and_decisions(monkeypatch):
    monkeypatch.delenv("VA__DIALOGUE__DECISION_BACKEND", raising=False)
    assert rc.pick_backend("local", "arm64") == "local"
    cmd = rc.worker_cmd(0, "local", "postgresql://db", Path("/x/site"), 8765, 16, None)
    assert cmd[cmd.index("--backend") + 1] == "local_server"
    env = rc.worker_env("local", "postgresql://db", "", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
    assert env["VA__DIALOGUE__DECISION_BACKEND"] == "local_server" and env["VA__LLM__BACKENDS__LOCAL_SERVER__MODEL"] == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    assert "VA__LLM__BACKENDS__LOCAL_SERVER__MODEL" not in rc.worker_env("local", "postgresql://db")


def test_the_env_override_reaches_the_local_server_entry():
    from dealer_core.config import load_config

    cfg = load_config(environ={"VA__LLM__BACKENDS__LOCAL_SERVER__MODEL": "some/model"})
    b = cfg.llm.backends["local_server"]
    assert b.model == "some/model" and b.runtime == "openai" and b.host == "http://127.0.0.1:8003/v1" and b.prompt_mode == "static"


def test_a_missing_local_server_says_what_to_start():
    model, why = rc.local_server_model("http://127.0.0.1:9/v1")                  # nothing listens on port 9
    assert model == "" and "mlx_lm.server" in why and "llama-server" in why and "8003" in why


def test_a_dry_run_with_a_local_server_prints_the_plan(tmp_path):
    (tmp_path / "data.js").write_text("// cars")
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_cluster.py"), "--dry-run", "--backend", "local", "--site", str(tmp_path)],
                         capture_output=True, text=True, timeout=60).stdout
    assert "backend   local" in out and "local_server" in out and "VA__DIALOGUE__DECISION_BACKEND=local_server" in out and "dealer_core.modelserver" not in out


def test_a_bedrock_worker_runs_as_the_owners_live_calls_do():
    """ARCH_REVIEW section 99: hands-free, the full quality, Nemotron on Bedrock for the decisions and the replies, no prefix mode, the shared call state and a worker id."""
    assert rc.pick_backend("bedrock", "arm64") == "bedrock"
    cmd = rc.worker_cmd(1, "bedrock", "postgresql://db", Path("/s/site"), 8785, 16, None)
    assert cmd[cmd.index("--backend") + 1] == "bedrock_nemotron" and "--hands-free" in cmd and "--prefix" not in cmd
    assert cmd[cmd.index("--quality") + 1] == "full" and cmd[cmd.index("--port") + 1] == "8786" and cmd[cmd.index("--worker-id") + 1] == "w2"
    assert cmd[cmd.index("--state-url") + 1] == "postgresql://db" and cmd[cmd.index("--max-calls") + 1] == "16"
    assert rc.worker_cmd(0, "bedrock", "x", Path("/s"), 8785, 4, "prefix")[rc.worker_cmd(0, "bedrock", "x", Path("/s"), 8785, 4, "prefix").index("--quality") + 1] == "prefix"
    env = rc.worker_env("bedrock", "postgresql://db", extra={"VA__VOICE__HOLD_REPLY": "true"})
    assert env["VOICE_AGENT_ALLOW_CLOUD"] == "1" and env["VA__DIALOGUE__DECISION_BACKEND"] == "bedrock_nemotron" and env["VA__LLM__FALLBACKS"] == "[]"
    assert env["VA__VOICE__HOLD_REPLY"] == "true" and env["DATABASE_URL"] == "postgresql://db"
    assert "VOICE_AGENT_ALLOW_CLOUD" not in {k for k in rc.worker_env("ollama", "x") if k == "VOICE_AGENT_ALLOW_CLOUD" and k not in __import__("os").environ}   # only the Bedrock backend sets it


def test_set_makes_environment_for_every_worker_and_refuses_a_bad_one():
    import pytest

    assert rc.parse_sets(["VA__VOICE__A=1", "B=x=y"]) == {"VA__VOICE__A": "1", "B": "x=y"} and rc.parse_sets(None) == {}
    with pytest.raises(SystemExit):
        rc.parse_sets(["nothing"])
    with pytest.raises(SystemExit):
        rc.parse_sets(["=1"])


def test_a_dry_run_of_the_bedrock_cluster_shows_the_settings_and_starts_nothing(tmp_path):
    (tmp_path / "data.js").write_text("// cars")
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_cluster.py"), "--dry-run", "--backend", "bedrock", "--workers", "2", "--site", str(tmp_path),
                          "--set", "VA__VOICE__HOLD_REPLY=true"], capture_output=True, text=True, timeout=60).stdout
    assert "backend   bedrock" in out and out.count("dealer_core.server") == 2 and "VA__VOICE__HOLD_REPLY=true" in out and "--hands-free" in out
    assert "dealer_core.modelserver" not in out and "dealer_core.gateway" in out


def test_redis_holds_the_load_reports_and_call_snapshots_by_default_and_the_database_only_the_stock_and_bookings(tmp_path):
    """5 Nov: the router's list of workers and the call snapshots in Redis (the default), bookings and stock in PostgreSQL."""
    (tmp_path / "data.js").write_text("// cars")
    cmd = rc.worker_cmd(0, "bedrock", "postgresql://db", Path("/s/site"), 8800, 16, None, "redis://127.0.0.1:6390/0")
    assert cmd[cmd.index("--state-url") + 1] == "redis://127.0.0.1:6390/0"                                      # the workers report there
    assert rc.worker_env("bedrock", "postgresql://db")["DATABASE_URL"] == "postgresql://db"                     # and the stock and bookings are still in the database
    assert rc.worker_cmd(0, "bedrock", "postgresql://db", Path("/s/site"), 8800, 16, None)[cmd.index("--state-url") + 1] == "postgresql://db"     # no state address: the database
    base = [sys.executable, str(ROOT / "scripts" / "run_cluster.py"), "--dry-run", "--backend", "bedrock", "--workers", "2", "--site", str(tmp_path)]
    default = subprocess.run(base, capture_output=True, text=True, timeout=60).stdout
    assert default.count("--state-url redis://127.0.0.1:6390/0") == 3 and "redis-server" in default                # 2 workers and the router, and a local Redis is started
    own = subprocess.run(base + ["--redis-url", "redis://cache:6379/0"], capture_output=True, text=True, timeout=60).stdout
    assert own.count("--state-url redis://cache:6379/0") == 3 and "redis-server" not in own                       # one that exists is not started
    old = subprocess.run(base + ["--no-redis"], capture_output=True, text=True, timeout=60).stdout
    assert "redis://" not in old and "redis-server" not in old and old.count("--state-url") == 3                                           # back in the database


def test_the_local_redis_keeps_its_data_on_disk_so_conversations_survive_a_restart(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda name: "/opt/homebrew/bin/redis-server" if name == "redis-server" else None)
    cmd = rc.redis_server_cmd(6391, tmp_path)
    assert cmd[0].endswith("redis-server") and cmd[cmd.index("--port") + 1] == "6391" and cmd[cmd.index("--bind") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--appendonly") + 1] == "yes" and cmd[cmd.index("--appendfsync") + 1] == "everysec" and cmd[cmd.index("--dir") + 1] == str(tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert rc.redis_server_cmd() is None
