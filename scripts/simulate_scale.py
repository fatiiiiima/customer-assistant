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

from dealer_core.sim.scale import run, scenarios  # noqa: E402

COLS = ["scenario", "protection", "calls", "busy %", "answer p50", "answer p90", "answer p99", "answers > 1 s %", "answers > 2 s %",
        "model full/prefix/code %", "shed %", "workers max", "model peak in flight"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeline", action="store_true")
    ap.add_argument("--only", help="run the scenarios whose name contains this")
    a = ap.parse_args()
    rows = []
    for adaptive in (True, False):
        for sc in scenarios(adaptive):
            if a.only and a.only not in sc.name:
                continue
            t0 = time.perf_counter()
            r = run(sc)
            r["_s"] = time.perf_counter() - t0
            rows.append(r)
            if a.timeline:
                print(f"\n{r['scenario']} (protection {r['protection']}): t / callers wanting / in calls / workers / answer p90 last 30 s")
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
