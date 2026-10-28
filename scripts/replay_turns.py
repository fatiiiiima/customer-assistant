r"""Replays recorded calls through BOTH model paths of the agent against the SAME backend, and says where they choose differently (Phase 2).

  python scripts/replay_turns.py --backend ollama                     # every words log in logs/audio_dump, newest first
  python scripts/replay_turns.py --backend gpu_server --first-sentence  # on the production server: also time each path to the first spoken sentence

Material: the words logs `logs/audio_dump/*_in.txt` (one line per event: time, kind, text; the caller's turns are the `heard` lines). Each call is replayed turn by turn through the
real agent engine, so the session (the car in focus, the cars shown, the booking under way) is what the code makes of the caller's words. The agent's own recorded replies are not used.

Per turn, `_fast_decide` (the code's cheap routing) comes first, exactly as in a call. When it is sure, no model is asked: both paths take that action by construction (route "code").
When it is not, the model is asked twice, from the same session state:
  two-call   AgentEngine._decide         the decision alone (the path that runs today)
  one-call   AgentEngine._decide_single  the action, the car and the reply in one structured request (dialogue.single_call)
and the script prints the action each chose, whether they agree (the car counts only for actions about one car), and the time each took. The one-call time is the time to the structured
answer: the whole object, reply included (it is not streamed). The session then moves on with the two-call decision, so both paths always see the same state.

`--first-sentence` also runs the whole turn both ways on copies of the session at the full level (the model writes) and times the first spoken sentence, which is what a caller waits for:
two-call = decision + the writer's first sentence, one-call = the one request (+ the writer only when the checks refuse its reply).

The times are the times of whatever machine and backend this runs on. On a laptop they are laptop times. Nothing here is a production number. Caller words are printed with long digit runs masked.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_PORTAL_DATA = ROOT.parent / "bmw-portal" / "site" / "data.js"
DEFAULT_LOGS = ROOT / "logs" / "audio_dump"
_LONG_NUMBER = re.compile(r"\d(?:[\s,.\-]*\d){4,}")


def read_words_log(path: Path) -> list[str]:
    """The caller's turns of one recorded call: the `heard` lines (what the engine was given after the recogniser and its guards)."""
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3 and parts[1] == "heard" and parts[2].strip():
            out.append(parts[2].strip())
    return out


def mask(text: str) -> str:
    """Phone numbers and other long digit runs are not printed."""
    return _LONG_NUMBER.sub("<number>", text)


def prefix_share(previous: str, current: str) -> float:
    """The share of `current` (in characters) that starts the same as `previous`: what a prefix cache could reuse. Characters, not tokens (vLLM caches whole blocks of 16 tokens)."""
    if not current:
        return 0.0
    return len(os.path.commonprefix([previous, current])) / len(current)


def pct(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))] if sorted_values else float("nan")


def decision_prompts(engine, s, text: str) -> tuple[str, str]:
    """The text of the two prompts that would be sent for this turn from this state: the two-call decision's (system + user) and the single call's."""
    from dealer_core.agent.engine import DECIDE_SYSTEM

    found, ids = engine._code_hint(text)
    two = DECIDE_SYSTEM + engine._context(s, text, (found, ids)) + f"CUSTOMER (now): {text}\nTOOL:"
    one = engine._single_system() + engine._single_user(s, text, found, ids)
    return two, one


def time_first_sentence(engine, s, text: str, single: bool) -> dict:
    """The whole turn at the full level on a COPY of the session; the time to the first spoken sentence and what was said."""
    from dealer_core.agent.governor import QualityGovernor

    saved = engine.single_call, engine.governor, engine.openers
    engine.single_call, engine.governor, engine.openers = single, QualityGovernor("full"), False        # (an opener is pre-rendered audio: it would hide the model's time)
    sess = copy.deepcopy(s)
    t0 = time.perf_counter()
    first, spoken, result = None, [], None
    try:
        for item in engine.handle_stream(sess, text):
            if isinstance(item, str):
                first = (time.perf_counter() - t0) * 1000 if first is None else first
                spoken.append(item)
            else:
                result = item
    finally:
        engine.single_call, engine.governor, engine.openers = saved
    return {"first_ms": first, "total_ms": (time.perf_counter() - t0) * 1000, "intent": result.debug.intent if result else "", "note": result.debug.note if result else "",
            "reply": " ".join(spoken)}


def replay_call(engine, texts: list[str], first_sentence: bool = False, alternate_from: int = 0, log=print) -> list[dict]:
    """Replay one call. One row per caller turn; `route` is "code" (the code's routing was sure) or "model" (both paths were asked)."""
    from dealer_core.agent.governor import QualityGovernor
    from dealer_core.agent.turn_schema import same_action

    s = engine.new_session()
    engine.opening(s)
    rows: list[dict] = []
    asked = alternate_from
    previous_prompts: tuple[str, str] | None = None
    for i, text in enumerate(texts):
        row: dict = {"turn": i, "words": mask(text), "route": "code"}
        before = copy.deepcopy(s)
        measured: dict = {}
        real_decide = engine._decide

        def reach_model(sess, t, _before=before, _row=row):
            """Called where the engine would ask the model to decide: measure both paths from the state the turn began in, hand the two-call decision back."""
            nonlocal asked, previous_prompts
            if "two" in measured:
                return measured["two"]
            order = ("two", "one") if asked % 2 == 0 else ("one", "two")           # which goes first alternates: a server's cache would otherwise favour one of them
            asked += 1
            for which in order:
                t0 = time.perf_counter()
                d = real_decide(_before, t) if which == "two" else engine._decide_single(_before, t)
                measured[which], measured[which + "_ms"] = d, (time.perf_counter() - t0) * 1000
            if first_sentence:
                measured["fs_two"] = time_first_sentence(engine, _before, t, False)
                measured["fs_one"] = time_first_sentence(engine, _before, t, True)
            prompts = decision_prompts(engine, _before, t)
            if previous_prompts is not None:
                _row["share_two"], _row["share_one"] = prefix_share(previous_prompts[0], prompts[0]), prefix_share(previous_prompts[1], prompts[1])
            previous_prompts = prompts
            return measured["two"]

        saved = engine.single_call, engine.governor, engine.openers
        engine.single_call, engine.governor, engine.openers = False, QualityGovernor("code"), False       # the session moves on at the code level: no reply is written by the model
        engine._decide = reach_model
        try:
            list(engine.handle_stream(s, text))
        finally:
            del engine._decide
            engine.single_call, engine.governor, engine.openers = saved
        if "two" in measured:
            two, one = measured["two"], measured["one"]
            row.update(route="model", two=(two.get("tool"), two.get("car_id")), one=(one.get("tool"), one.get("car_id")), two_ms=measured["two_ms"], one_ms=measured["one_ms"],
                       two_error=two.get("error"), one_error=one.get("error"), one_note=one.get("single", {}),
                       say=bool(one.get("say")), say_chars=len(one.get("say") or ""))
            failed = bool(row["two_error"] or row["one_error"])
            row["agree"] = same_action(row["two"], row["one"], before.car) and not failed              # a decision that names no car means the car in focus, as in the engine
            row["agree_strict"] = same_action(row["two"], row["one"]) and not failed                    # ...and with the car compared as written
            if first_sentence:
                row["fs_two"], row["fs_one"] = measured["fs_two"], measured["fs_one"]
        rows.append(row)
        log(format_row(len(rows), row))
    return rows


def format_row(n: int, r: dict) -> str:
    if r["route"] == "code":
        return f"  {r['turn']:2d}  code   {'-':26s} {'-':26s}   -        -        -  {r['words'][:60]}"
    fmt = lambda a: f"{a[0]}{' ' + a[1] if a[1] else ''}"
    flag = "yes" if r["agree"] else "NO "
    err = f" [two-call {r['two_error']}]" if r.get("two_error") else ""
    err += f" [one-call {r['one_error']}]" if r.get("one_error") else ""
    note = r["one_note"].get("malformed")
    return (f"  {r['turn']:2d}  model  {fmt(r['two']):26s} {fmt(r['one']):26s}  {flag}  {r['two_ms']:6.0f}   {r['one_ms']:6.0f}  {r['words'][:60]}"
            + err + (f" [malformed: {note}]" if note else ""))


def summarise(rows: list[dict], decide_deadline_ms: float = 1200.0, single_deadline_ms: float = 1500.0) -> dict:
    model = [r for r in rows if r["route"] == "model"]
    agree = [r for r in model if r["agree"]]
    two_ms = sorted(r["two_ms"] for r in model)
    one_ms = sorted(r["one_ms"] for r in model)
    out = {
        "turns": len(rows), "routed_by_code": len(rows) - len(model), "decided_by_the_model": len(model), "agree_on_model_turns": len(agree),
        "agree_share_model_turns": (len(agree) / len(model)) if model else None,
        "agree_strict_on_model_turns": sum(1 for r in model if r["agree_strict"]),
        "agree_share_all_turns": ((len(rows) - len(model) + len(agree)) / len(rows)) if rows else None,
        "disagreements": [{"turn": r["turn"], "words": r["words"], "two_call": r["two"], "one_call": r["one"], "two_error": r.get("two_error"), "one_error": r.get("one_error"),
                           "malformed": r["one_note"].get("malformed")} for r in model if not r["agree"]],
        "malformed": sum(1 for r in model if r["one_note"].get("malformed")), "one_call_errors": sum(1 for r in model if r.get("one_error")),
        "two_call_errors": sum(1 for r in model if r.get("two_error")),
        "two_call_ms": {"median": st.median(two_ms) if two_ms else None, "p90": pct(two_ms, 0.9) if two_ms else None, "max": two_ms[-1] if two_ms else None,
                        f"over_{decide_deadline_ms:.0f}": sum(1 for v in two_ms if v > decide_deadline_ms)},
        "one_call_ms": {"median": st.median(one_ms) if one_ms else None, "p90": pct(one_ms, 0.9) if one_ms else None, "max": one_ms[-1] if one_ms else None,
                        f"over_{single_deadline_ms:.0f}": sum(1 for v in one_ms if v > single_deadline_ms)},
        "one_call_gave_a_reply": sum(1 for r in model if r.get("say")),
    }
    shares = lambda key: [r[key] for r in rows if key in r]
    if shares("share_two"):
        out["prefix_share_chars"] = {"two_call_prompt": st.mean(shares("share_two")), "one_call_prompt": st.mean(shares("share_one")), "pairs": len(shares("share_two"))}
    fs = [r for r in model if "fs_two" in r]
    if fs:
        a = sorted(r["fs_two"]["first_ms"] for r in fs if r["fs_two"]["first_ms"] is not None)
        b = sorted(r["fs_one"]["first_ms"] for r in fs if r["fs_one"]["first_ms"] is not None)
        out["first_sentence_ms"] = {"turns": len(fs), "two_call": {"median": st.median(a) if a else None, "p90": pct(a, 0.9) if a else None},
                                    "one_call": {"median": st.median(b) if b else None, "p90": pct(b, 0.9) if b else None},
                                    "one_call_reply_used": sum(1 for r in fs if "single call: its reply was used" in r["fs_one"]["note"]),
                                    "one_call_reply_refused": sum(1 for r in fs if "single call: its reply was refused" in r["fs_one"]["note"]),
                                    "same_spoken_intent": sum(1 for r in fs if r["fs_two"]["intent"] == r["fs_one"]["intent"])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="ollama", help="a name from config llm.backends, used for the decisions AND the replies")
    ap.add_argument("--logs", type=Path, default=DEFAULT_LOGS, help="the folder with the *_in.txt words logs (default logs/audio_dump)")
    ap.add_argument("--portal-data", type=Path, default=DEFAULT_PORTAL_DATA, help="the website's site/data.js (the cars the recorded calls were about)")
    ap.add_argument("--calls", type=int, default=0, help="only the newest N calls (default: all)")
    ap.add_argument("--first-sentence", action="store_true", help="also time the whole turn both ways to the first spoken sentence (twice the model calls)")
    ap.add_argument("--deadline-s", type=float, default=10.0, help="how long either path may take before it counts as no answer. Generous on purpose, so agreement is not decided by speed; "
                                                                  "the production deadlines are counted separately (latency.decide_deadline_ms, latency.first_sentence_deadline_ms)")
    ap.add_argument("--json", type=Path, default=ROOT / "logs" / "replay_turns.json")
    a = ap.parse_args()

    from dealer_core.app import build_app
    from dealer_core.cli.chat import start
    from dealer_core.config import load_config

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml", environ=dict(os.environ))       # (the .env file is not read: the backend is the one named here)
    cfg.dialogue.decision_backend = a.backend
    cfg.llm.fallbacks = []                                                                        # a failure shows up as one; no other runtime is loaded in its place
    app = build_app(cfg, inventory="portal", portal_data=a.portal_data, engine="agent", backend=a.backend)
    start(app, quiet=True)
    engine = app.engine
    engine.decide_deadline_s = engine.single_deadline_s = a.deadline_s
    if engine.reply_writer is not None:
        engine.reply_writer.first_deadline_s, engine.reply_writer.deadline_s = a.deadline_s, a.deadline_s
    logs = sorted(a.logs.glob("*_in.txt"), reverse=True)                                          # newest first
    if a.calls:
        logs = logs[:a.calls]
    print(f"backend {a.backend} ({cfg.llm.backends[a.backend].model}) on this machine; {len(logs)} recorded calls; deadlines for the replay {a.deadline_s:.0f} s\n", flush=True)
    engine._decide(engine.new_session(), "hello")                                                  # warm up (the first request after a model load is slow)
    all_rows: list[dict] = []
    for path in logs:
        texts = read_words_log(path)
        print(f"{path.name}: {len(texts)} caller turns\n  turn route  two-call action             one-call action           agree  two ms   one ms  words", flush=True)
        for r in replay_call(engine, texts, a.first_sentence, alternate_from=len(all_rows)):
            r["call"] = path.name
            all_rows.append(r)
        print()
    summary = summarise(all_rows, cfg.latency.decide_deadline_ms, cfg.latency.first_sentence_deadline_ms)
    print(json.dumps(summary, indent=1, default=str))
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps({"backend": a.backend, "model": cfg.llm.backends[a.backend].model, "summary": summary, "rows": all_rows}, indent=1, default=str), encoding="utf-8")
    print(f"\nper-turn rows written to {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
