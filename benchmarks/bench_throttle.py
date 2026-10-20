"""Detect thermal / power throttling: sustained all-core load, sampled once per second.

Reports throughput (GFLOP/s of a fixed matmul workload), CPU clock, and thermal-zone temperature
over time. Throttling shows as throughput and clock falling while the work stays identical.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).parent))
from common import save_result  # noqa: E402


def temps_c() -> list[float]:
    """Thermal-zone temperatures via Windows performance counters (no admin needed). Kelvin -> C."""
    cmd = "(Get-Counter '\\Thermal Zone Information(*)\\Temperature' -ErrorAction SilentlyContinue).CounterSamples | ForEach-Object { $_.CookedValue }"
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=20).stdout.split()
        return [round(float(x) - 273.15, 1) for x in out]
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=150)
    ap.add_argument("--threads", type=int, default=1, help="python worker threads; BLAS already uses all cores per matmul")
    a = ap.parse_args()
    stop = threading.Event()
    done = [0] * a.threads
    m = 1280  # large enough that one matmul keeps all cores busy; avoids BLAS x Python thread oversubscription

    def worker(i: int) -> None:
        x = np.random.rand(m, m).astype(np.float32)
        while not stop.is_set():
            _ = x @ x
            done[i] += 1

    ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(a.threads)]
    batt = psutil.sensors_battery()
    info = {"threads": a.threads, "on_ac_power": None if batt is None else batt.power_plugged,
            "battery_percent": None if batt is None else batt.percent, "temps_start_c": temps_c()}
    flop_per = 2 * m**3
    samples, last, t0 = [], 0, time.perf_counter()
    for t in ts:
        t.start()
    for s in range(a.seconds):
        time.sleep(1)
        tot = sum(done)
        gf = (tot - last) * flop_per / 1e9
        last = tot
        f = psutil.cpu_freq()
        samples.append({"t": s + 1, "gflops": round(gf, 1), "mhz": round(f.current) if f else None})
        if s in (a.seconds // 2,):
            info["temps_mid_c"] = temps_c()
    stop.set()
    info["temps_end_c"] = temps_c()
    g = [x["gflops"] for x in samples]
    burst = float(np.mean(g[2:20]))
    tail = g[30:] if len(g) > 40 else g[len(g) // 2:]
    sustained = float(np.median(tail))
    info.update(burst_gflops_first_20s=round(burst, 1), sustained_gflops_median_after_30s=round(sustained, 1),
                sustained_vs_burst=round(sustained / burst, 2),
                p5_gflops=round(float(np.percentile(g[5:], 5)), 1), p95_gflops=round(float(np.percentile(g[5:], 95)), 1),
                seconds_below_half_of_burst=sum(1 for x in g if x < burst / 2),
                note="sustained_vs_burst well below 1.0 means power/thermal limiting after the initial burst; "
                     "psutil MHz on Windows is the base clock and is NOT a real-time frequency")
    print({k: v for k, v in info.items()})
    save_result("throttle", {"summary": info, "samples": samples, "wall_s": round(time.perf_counter() - t0, 1)})


if __name__ == "__main__":
    main()
