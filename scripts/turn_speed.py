"""Text turns through the real agent engine, set up as the worker runs it for the Nemotron-only test (voice.model_only), timing what the caller waits for:
the first words said and the whole turn, and whether the action was the expected one. One run can compare the settings side by side (ARCH_REVIEW section 67):
  baseline  the decision, then the reply (two calls)
  parallel  the decision while the replies for the likely lookups are written (voice.parallel)
  one_call  the action and the reply in one call (voice.one_call)

  uv run python scripts/turn_speed.py --backend gpu_server                       (the local model server on port 8003: free)
  AWS_BEARER_TOKEN_BEDROCK=... VOICE_AGENT_ALLOW_CLOUD=1 uv run python scripts/turn_speed.py --backend bedrock_nemotron --modes baseline parallel --repeat 2
      This COSTS MONEY (--dry-run says how many calls). The key is read from the environment and never printed or saved.
The same conversations as scripts/bedrock_speed.py (24 turns, 17 with an expected action). Each turn is sent after the agent's own earlier replies, as in a call.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bedrock_speed as bs

MODES = {"baseline": {}, "parallel": {"VA__VOICE__PARALLEL": "true"}, "one_call": {"VA__VOICE__ONE_CALL": "true"}}
KEY_ENV = "AWS_BEARER_TOKEN_BEDROCK"
TOKENS_DECIDE, TOKENS_REPLY = 600, 1100              # input tokens of one decision / one reply prompt (about 2,400 / 4,400 characters, section 62), an estimate
CALLS = {"baseline": (1, 1), "parallel": (1, 5), "one_call": (0, 1)}      # (decisions, replies) a turn, at most: a parallel turn writes a reply for up to 5 lookups


def pct(xs: list[float], p: int) -> int | None:
    xs = sorted(xs)
    return round(xs[max(0, -(-p * len(xs) // 100) - 1)]) if xs else None


def conversations() -> list[list[tuple[str, str | None]]]:
    from compare_agent_models import CONVERSATIONS

    return [list(c) for c in CONVERSATIONS + bs.OWNER_CALLS]


def make_engine(backend: str, mode: str):
    from dealer_core.app import build_app
    from dealer_core.cli.chat import start
    from dealer_core.config import load_config
    from dealer_core.voice.pipeline import apply_voice_profile

    env = {"VA__DIALOGUE__DECISION_BACKEND": backend, "VA__LLM__FALLBACKS": "[]", "VA__VOICE__MODEL_ONLY": "true", "VA__VOICE__SILENT_TURN_FALLBACK": "false", **MODES[mode]}
    if backend == "gpu_server":
        env.update(bs.LOCAL, VA__VOICE__DECIDE_DEADLINE_MS="30000")              # the local server answers one request at a time: a parallel decision would otherwise time out
    cfg = load_config(environ=env, overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=bs.SITE_DATA, backend=backend, engine="agent", quality="full")
    start(app, quiet=True)
    apply_voice_profile(app, cfg)
    return app.engine


def run_mode(backend: str, mode: str, repeat: int, say=print) -> dict:
    rows = []
    for rep in range(repeat):
        e = make_engine(backend, mode)
        for convo in conversations():
            s = e.new_session()
            e.opening(s)
            for line, want in convo:
                t0, first, last = time.perf_counter(), None, None
                for last in e.handle_stream(s, line):
                    if first is None and isinstance(last, str) and last.strip():
                        first = (time.perf_counter() - t0) * 1000
                d = last.debug
                rows.append({"repeat": rep, "customer": line, "want": want, "intent": d.intent, "first_ms": None if first is None else round(first),
                             "total_ms": round((time.perf_counter() - t0) * 1000), "reply": last.reply[:200],
                             "ahead": "replies written ahead" in d.note, "ahead_used": " was used" in d.note, "note": d.note[:600]})
    first = [r["first_ms"] for r in rows if r["first_ms"] is not None]
    judged = [r for r in rows if r["want"]]
    out = {"mode": mode, "turns": len(rows), "silent": sum(r["first_ms"] is None for r in rows), "first_p50": pct(first, 50), "first_p90": pct(first, 90),
           "total_p50": pct([r["total_ms"] for r in rows], 50), "total_p90": pct([r["total_ms"] for r in rows], 90),
           "right": f"{sum(r['intent'] == r['want'] for r in judged)}/{len(judged)}", "ahead": sum(r["ahead"] for r in rows), "ahead_used": sum(r["ahead_used"] for r in rows),
           "rows": rows}
    say(f"{mode:9s} first words p50 {out['first_p50']} / p90 {out['first_p90']} ms; turn p50 {out['total_p50']} / p90 {out['total_p90']} ms; right action {out['right']}; "
        f"silent {out['silent']}" + (f"; reply ready for the decision {out['ahead_used']}/{out['ahead']}" if out["ahead"] else ""))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="gpu_server", help="gpu_server (local, free) or bedrock_nemotron (costs money)")
    p.add_argument("--modes", nargs="+", choices=list(MODES), default=["baseline", "parallel"])
    p.add_argument("--repeat", type=int, default=1, help="how many times the conversations are run for each mode")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    turns = sum(len(c) for c in conversations()) * a.repeat
    cloud = a.backend.startswith("bedrock")
    dec = sum(CALLS[m][0] for m in a.modes) * turns
    rep = sum(CALLS[m][1] for m in a.modes) * turns
    print(f"{len(a.modes)} mode(s) x {turns} turns on {a.backend}: at most {dec + rep} model calls, about {(dec * TOKENS_DECIDE + rep * TOKENS_REPLY):,} input tokens "
          f"(an estimate: {TOKENS_DECIDE} for a decision, {TOKENS_REPLY} for a reply)")
    if a.dry_run:
        print("dry run: nothing sent")
        return 0
    if cloud and (not os.environ.get(KEY_ENV) or os.environ.get(bs.ALLOW_ENV) != "1"):
        print(f"not sent: set {KEY_ENV} (your key) and {bs.ALLOW_ENV}=1 in your own terminal first. This run costs money.")
        return 2
    results = [run_mode(a.backend, m, a.repeat) for m in a.modes]
    out = ROOT / "benchmarks" / "results" / f"turn_speed_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backend": a.backend, "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
