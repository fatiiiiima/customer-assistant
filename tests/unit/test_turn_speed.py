"""scripts/turn_speed.py (ARCH_REVIEW section 67): text turns through the engine, timed. With a scripted engine: nothing leaves this machine."""
import importlib.util
from pathlib import Path

from .test_model_only import parallel

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("turn_speed", ROOT / "scripts" / "turn_speed.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)


def test_a_cloud_run_is_refused_without_the_key_and_the_switch_and_a_dry_run_counts_the_calls(monkeypatch, capsys):
    monkeypatch.delenv(ts.KEY_ENV, raising=False)
    monkeypatch.delenv(ts.bs.ALLOW_ENV, raising=False)
    assert ts.main(["--backend", "bedrock_nemotron"]) == 2 and "costs money" in capsys.readouterr().out
    assert ts.main(["--backend", "bedrock_nemotron", "--modes", "baseline", "parallel", "--dry-run"]) == 0
    out = capsys.readouterr().out
    turns = sum(len(c) for c in ts.conversations())
    assert f"2 mode(s) x {turns} turns" in out and f"at most {turns * 2 + turns * 6} model calls" in out and "nothing sent" in out      # 2 for the baseline, up to 6 for the parallel


def test_each_turn_is_timed_to_its_first_words_and_scored_against_the_expected_action(monkeypatch, tmp_path):
    engines = []

    def make(backend, mode):
        e, _ = parallel(["search_cars", "get_car"])
        e.parallel = mode == "parallel"
        engines.append(e)
        return e

    monkeypatch.setattr(ts, "make_engine", make)
    monkeypatch.setattr(ts, "conversations", lambda: [[("do you have any Toyotas", "search_cars"), ("tell me about the first one", "search_cars"), ("hello", None)]])
    said = []
    for mode in ("baseline", "parallel"):
        r = ts.run_mode("gpu_server", mode, 1, say=said.append)
        assert r["turns"] == 3 and r["silent"] == 0 and r["first_p50"] is not None and r["first_p50"] <= r["total_p50"]
        assert r["right"] == "1/2"                                                   # judged only where an action is expected: the second turn picked get_car
    assert r["ahead"] == 3 and r["ahead_used"] >= 2 and "reply ready for the decision" in said[1] and "reply ready" not in said[0]
