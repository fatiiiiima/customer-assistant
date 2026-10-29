"""500+ callers on a simulated cluster: the real router, quality governor and measured call cap, against stand-in hardware.

    python scripts/simulate_scale.py                 # every scenario, protection on (measured cap, shedding, deadline) vs off
    python scripts/simulate_scale.py --timeline      # also how each run evolved, every 30 s

See src/dealer_core/sim/scale.py for what is real and what is a stand-in.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dealer_core.sim.scale import model_capacity_sweep, run, scenarios  # noqa: E402

COLS = ["scenario", "protection", "calls", "busy %", "answer p50", "answer p90", "answer p99", "answers > 1 s %", "answers > 2 s %",
        "model full/prefix/code %", "shed %", "workers max", "model peak in flight", "prefix while model idle %", "workers held down while model idle %", "probe requests", "probes while model overloaded", "seconds to recover"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeline", action="store_true")
    ap.add_argument("--model-capacity", action="store_true", help="instead of the scenarios: how much model capacity (requests served together at full speed) 500, 1,000 and 1,500 callers need")
    ap.add_argument("--only", help="run the scenarios whose name contains this")
    ap.add_argument("--old-probe", action="store_true", help="the probe rule before B5: only when no turn is using the model, never skipped (to compare)")
    ap.add_argument("--probe-max-in-flight", type=int, default=None, help="probe when at most this many turns are using the model (default: dialogue.probe_max_in_flight)")
    ap.add_argument("--probe-no-skip", action="store_true", help="do not skip a probe when a caller's model-written turn timed the model recently")
    a = ap.parse_args()
    if a.model_capacity:
        for callers, workers in ((500, 72), (1000, 120), (1500, 180)):
            print(f"\n{callers} callers (up to {workers} workers; stand-in speeds: decision 120 ms, first sentence 300 ms, 60% of turns answered by code)")
            print("| model slots | busy % | answer p90 | answers > 2 s % | model full/prefix/code % | shed % | model peak in flight | workers max |\n|---|---|---|---|---|---|---|---|")
            for r in model_capacity_sweep(callers, [10, 20, 30, 45, 60, 90, 160], workers):
                print(f"| {r['model slots']} | {r['busy %']} | {r['answer p90']} | {r['answers > 2 s %']} | {r['model full/prefix/code %']} | {r['shed %']} | {r['model peak in flight']} | {r['workers max']} |")
        return 0
    rows = []
    for adaptive in (True, False):
        for sc in scenarios(adaptive):
            if a.only and a.only not in sc.name:
                continue
            if a.old_probe:
                sc.probe_max_in_flight, sc.probe_skip_fresh = 0, False
            if a.probe_max_in_flight is not None:
                sc.probe_max_in_flight = a.probe_max_in_flight
            if a.probe_no_skip:
                sc.probe_skip_fresh = False
            t0 = time.perf_counter()
            r = run(sc)
            r["_s"] = time.perf_counter() - t0
            rows.append(r)
            if a.timeline:
                print(f"\n{r['scenario']} (protection {r['protection']}): t / callers wanting / in calls / workers / answer p90 last 30 s / workers above their model budget / model load")
                for row in r["timeline"]:
                    print("   ", row)
    print("\n| " + " | ".join(COLS) + " |")
    print("|" + "---|" * len(COLS))
    for r in sorted(rows, key=lambda r: (r["scenario"], r["protection"])):
        print("| " + " | ".join(str(r[c]) for c in COLS) + " |")
    print(f"\n{len(rows)} runs, {sum(r['_s'] for r in rows):.0f} s of computing (10 simulated minutes each, measured after the 2-minute ramp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
