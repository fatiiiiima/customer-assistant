r"""Scenario replayer: runs scripted fake-customer conversations through the real engine and checks every turn.

  .venv\Scripts\python.exe -m dealer_core.cli.replay --backend openvino_gpu                  # the demo, one run each, transcripts shown
  .venv\Scripts\python.exe -m dealer_core.cli.replay --backend openvino_gpu --runs 20 --quiet # reliability: 20 runs per scenario

A fresh in-memory database and a fresh session are used for every run; the language model is loaded once and shared.
Exit code 1 if any expectation failed in any run.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import yaml

from ..app import ROOT, App, build_app, demo_clock
from ..config import Config, load_config
from ..nlu.backend import BackendError
from ..nlu.chain import BackendChain
from .chat import format_debug

DEFAULT_FILE = ROOT / "scenarios" / "demo.yaml"


def check(expect: dict, reply: str, dbg, ended: bool) -> list[str]:
    """Failed expectations for one turn (empty list = pass)."""
    bad: list[str] = []
    if "intent" in expect and dbg.intent != expect["intent"]:
        bad.append(f"intent {dbg.intent!r}, expected {expect['intent']!r}")
    if "via" in expect and not dbg.via.startswith(expect["via"]):
        bad.append(f"handled by {dbg.via!r}, expected {expect['via']!r}")
    if "state_after" in expect and dbg.state_after != expect["state_after"]:
        bad.append(f"state_after {dbg.state_after!r}, expected {expect['state_after']!r}")
    if "ended" in expect and ended != expect["ended"]:
        bad.append(f"ended={ended}, expected {expect['ended']}")
    for part in expect.get("reply_has", []):
        if part not in reply:
            bad.append(f"reply lacks {part!r}")
    for part in expect.get("reply_not", []):
        if part in reply:
            bad.append(f"reply contains {part!r}")
    for part in expect.get("tool_has", []):
        if not any(part in t for t in dbg.tools):
            bad.append(f"no tool call containing {part!r}")
    for part in expect.get("decision_has", []):
        if part not in dbg.decision:
            bad.append(f"policy decision lacks {part!r}")
    if dbg.violations:
        bad.append(f"voice-rule violations: {dbg.violations}")
    return bad


def run_once(app: App, scenario: dict, *, show: bool, debug: bool, pause: float) -> tuple[list[str], list[float]]:
    s = app.engine.new_session()
    failures: list[str] = []
    llm_ms: list[float] = []
    if show:
        print(f"Agent: {app.engine.opening(s)}")
    for n, step in enumerate(scenario["steps"], 1):
        if show:
            print(f"\nYou:   {step['say']}")
            if pause:
                time.sleep(pause)
        r = app.engine.handle(s, step["say"])
        if r.debug.via.startswith("llm"):
            llm_ms.append(r.debug.nlu_ms)
        bad = check(step.get("expect", {}), r.reply, r.debug, r.ended)
        failures += [f"turn {n} ({step['say']!r}): {b}" for b in bad]
        if show:
            print(f"Agent: {r.reply}" + ("   <-- FAILED" if bad else ""))
            if debug or bad:
                print(format_debug(r.debug))
            if r.ended:
                print(f"\n[{'transferred to a human (' + s.handoff.reason + ')' if s.handoff else 'call ended'}]")
            if pause:
                time.sleep(pause)
        if r.ended and n < len(scenario["steps"]):
            failures.append(f"turn {n}: the call ended before the script did")
            break
    return failures, llm_ms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, default=DEFAULT_FILE)
    ap.add_argument("--backend", help="llm backend from config; default llm.text_backend")
    ap.add_argument("--runs", type=int, default=1, help="runs per scenario (the first is shown unless --quiet)")
    ap.add_argument("--only", type=int, nargs="*", help="scenario numbers to run (1-based)")
    ap.add_argument("--hybrid", action="store_true", help="hybrid replies (model-written lead-in before low-risk templated answers)")
    ap.add_argument("--debug", action="store_true", help="show the per-turn trace for every turn")
    ap.add_argument("--quiet", action="store_true", help="no transcripts, only the summary and any failures")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to pause between turns (for a live demo)")
    ap.add_argument("--clock", choices=["demo", "real"], default="demo")
    a = ap.parse_args(argv)

    cfg: Config = load_config()
    scenarios = yaml.safe_load(a.file.read_text(encoding="utf-8"))
    if a.only:
        scenarios = [sc for i, sc in enumerate(scenarios, 1) if i in a.only]
    chain = BackendChain.from_config(cfg, primary=a.backend or cfg.llm.text_backend)
    t0 = time.perf_counter()
    try:
        print(chain.start())
        print(f"(model loaded and warmed up in {time.perf_counter() - t0:.1f} s)\n")
    except BackendError as e:
        print(f"No language-model backend available: {e}")
        return 2
    clock = demo_clock(cfg) if a.clock == "demo" else None
    all_ms: list[float] = []
    summary, failed_any = [], False
    for sc in scenarios:
        ok_runs, fail_msgs, ms = 0, [], []
        print("=" * 100 + f"\n{sc['name']}\n" + "=" * 100)
        for run in range(a.runs):
            app = build_app(cfg, chain=chain, clock=clock, lead_in=True if a.hybrid else None)
            f, llm_ms = run_once(app, sc, show=(run == 0 and not a.quiet) or False, debug=a.debug, pause=a.pause)
            ms += llm_ms
            if f:
                fail_msgs.append((run + 1, f))
            else:
                ok_runs += 1
        all_ms += ms
        summary.append((sc["name"], ok_runs, a.runs, ms))
        failed_any |= ok_runs < a.runs
        for run_no, f in fail_msgs[:3]:
            print(f"  run {run_no} FAILED:\n    " + "\n    ".join(f))
        print(f"\n  -> {ok_runs}/{a.runs} runs passed")
    print("\n" + "=" * 100 + "\nSUMMARY")
    for name, ok, n, ms in summary:
        lat = f"LLM turns: {len(ms)}, NLU p50 {statistics.median(ms):.0f} ms, max {max(ms):.0f} ms" if ms else "no LLM turns (all answered by rules)"
        print(f"  {'PASS' if ok == n else 'FAIL'}  {ok:3d}/{n:<3d}  {name}   [{lat}]")
    if all_ms:
        qs = statistics.quantiles(all_ms, n=20) if len(all_ms) > 1 else [all_ms[0]] * 19
        print(f"\n  all LLM turns: n={len(all_ms)}, p50 {statistics.median(all_ms):.0f} ms, p95 {qs[18]:.0f} ms, max {max(all_ms):.0f} ms")
    return 1 if failed_any else 0


if __name__ == "__main__":
    sys.exit(main())
