r"""Hold many calls open at once through the router, with a few real conversations on top (the "1000 callers" question, as far as one machine can answer it).

  .venv/bin/python scripts/soak_test.py --gateway http://127.0.0.1:8760 --callers 1000 --active 8 --ramp-secs 120 --hold-secs 60

What it does, in order:
  1. BASELINE   --active real conversations (the load test's hands-free callers) alone: the first-sound wait with nothing else connected.
  2. RAMP       --callers quiet callers join over --ramp-secs. Each asks the router for a worker (/assign), opens the WebSocket, says hello, hears its greeting and then keeps
                the line open, sending one 20 ms frame of quiet every --frame-every seconds. (A real open microphone sends 50 frames a second: that is what costs a worker
                its CPU, so a quiet line here is a LIGHT call. The result is for connected calls, not for 1000 people talking, and it says so.)
  3. HOLD       with all of them connected, the same --active conversations again: does the wait change while the rest are connected?
  4. END        every quiet caller closes.
Reported: how many connected, were refused ("busy") or failed; the router's answer time; the memory and processor each group of our processes used before, at the peak and
after; the real callers' wait before and during. Quiet callers are given their greeting and then nothing (start the workers with VA__VOICE__STILL_THERE_SECS and
VA__VOICE__HANG_UP_SECS very large, or the server hangs up on them after about 18 s).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import websockets  # noqa: E402
from load_test import GATEWAY, MIX, group_of, pct, staggered  # noqa: E402
from portal_call import FRAME, caller_audio  # noqa: E402

QUIET = np.zeros(FRAME, dtype=np.int16).tobytes()


def assign_url(gateway: str) -> tuple[str | None, float, str]:
    """(worker address or None, seconds it took, why not)."""
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(gateway.rstrip("/") + "/assign", timeout=10) as r:
            return json.loads(r.read())["url"], time.perf_counter() - t0, ""
    except Exception as e:  # noqa: BLE001 - counted, not raised: a refused or failed assignment is a result
        return None, time.perf_counter() - t0, type(e).__name__


class Quiet:
    """One caller with a quiet line."""

    def __init__(self, n: int, gateway: str, origin: str, frame_every: float) -> None:
        self.n, self.gateway, self.origin, self.frame_every = n, gateway, origin, frame_every
        self.status = "new"
        self.assign_s = self.connect_s = 0.0
        self.ready_s = 0.0
        self.heard_audio_s = 0.0
        self.stop = asyncio.Event()

    async def run(self) -> None:
        url, self.assign_s, why = await asyncio.to_thread(assign_url, self.gateway)
        if url is None:
            self.status = f"assign:{why}"
            return
        t0 = time.perf_counter()
        try:
            async with websockets.connect(url, max_size=None, origin=self.origin, open_timeout=30, ping_interval=None) as ws:
                self.connect_s = time.perf_counter() - t0
                await ws.send(json.dumps({"type": "hello", "car": None, "call_id": f"soak{self.n}"}))
                self.status = "open"

                async def reader() -> None:
                    async for m in ws:
                        if isinstance(m, bytes):
                            self.heard_audio_s += len(m) / 2 / 22050
                        else:
                            e = json.loads(m)
                            if e.get("type") == "ready" and not self.ready_s:
                                self.ready_s = time.perf_counter() - t0
                            elif e.get("type") == "busy":
                                self.status = "busy"
                task = asyncio.create_task(reader())
                while not self.stop.is_set() and not task.done():
                    await ws.send(QUIET)
                    try:
                        await asyncio.wait_for(self.stop.wait(), self.frame_every)
                    except asyncio.TimeoutError:
                        pass
                task.cancel()
                if self.status == "open":
                    self.status = "held" if not task.done() or task.cancelled() else "dropped"
        except Exception as e:  # noqa: BLE001
            self.status = f"error:{type(e).__name__}"


def processes() -> dict[str, tuple[float, float]]:
    """{group of ours: (resident memory in GB, cores in use over the last second)} for the worker, router, these callers, the stand-in services."""
    import psutil

    out: dict[str, list[float]] = {}
    for p in psutil.process_iter(["cmdline"]):
        text = " ".join(p.info["cmdline"] or [])
        g = group_of(p.info["cmdline"]) or ("stand-in services" if "stub_services" in text else "these callers (this script)" if "soak_test.py" in text
                                            else "database (PostgreSQL)" if "postgres" in text.lower() and "pgserver" in text.lower() else None)
        if not g:
            continue
        try:
            row = out.setdefault(g, [0.0, 0.0])
            row[0] += p.memory_info().rss / 1e9
            row[1] += p.cpu_percent(None) / 100.0
        except psutil.Error:
            pass
    return {k: (v[0], v[1]) for k, v in out.items()}


async def watch(into: list, every: float = 5.0) -> None:
    import psutil

    processes()                                                                  # starts the per-process clocks
    psutil.cpu_percent()
    while True:
        await asyncio.sleep(every)
        into.append({"t": time.time(), "machine_cpu": psutil.cpu_percent(), "machine_mem_used_gb": (psutil.virtual_memory().total - psutil.virtual_memory().available) / 1e9, "groups": processes()})


async def real_calls(gateway: str, origin: str, n: int, variants, tag: str) -> list[dict]:
    GATEWAY["url"] = gateway
    out = await asyncio.gather(*[staggered("", origin, variants[i % len(variants)], i, 0.25, "handsfree") for i in range(n)], return_exceptions=True)
    return [r for r in out if isinstance(r, dict)]


def waits(results: list[dict]) -> dict:
    xs = [t["silence_s"] for r in results for t in r["turns"] if t.get("silence_s") is not None]
    return {"calls": len(results), "ok": sum(1 for r in results if r["status"] == "ok"), "turns": len(xs), "median_s": round(st.median(xs), 2) if xs else None,
            "p90_s": round(pct(xs, 90), 2) if xs else None, "worst_s": round(max(xs), 2) if xs else None, "over_2s": round(sum(1 for x in xs if x > 2) / len(xs), 2) if xs else None}


def summarise(quiet: list[Quiet]) -> dict:
    by: dict[str, int] = {}
    for q in quiet:
        key = q.status.split(":")[0] if q.status.startswith(("held", "busy", "dropped")) else q.status
        by[key] = by.get(key, 0) + 1
    a = [q.assign_s for q in quiet if q.assign_s]
    c = [q.connect_s for q in quiet if q.connect_s]
    r = [q.ready_s for q in quiet if q.ready_s]
    f = lambda xs, p: round(float(np.percentile(xs, p)), 3) if xs else None  # noqa: E731
    return {"outcomes": dict(sorted(by.items())), "assign_s": {"median": f(a, 50), "p99": f(a, 99), "worst": f(a, 100)}, "connect_s": {"median": f(c, 50), "p99": f(c, 99), "worst": f(c, 100)},
            "ready_s": {"median": f(r, 50), "p99": f(r, 99), "worst": f(r, 100)}, "greeting_heard": sum(1 for q in quiet if q.heard_audio_s > 0.5)}


def worker_health(gateway: str) -> list[dict]:
    try:
        with urllib.request.urlopen(gateway.rstrip("/") + "/workers", timeout=5) as r:
            ws = json.loads(r.read()).get("workers", [])
    except Exception:  # noqa: BLE001
        return []
    out = []
    for w in ws:
        try:
            with urllib.request.urlopen(w["url"].replace("ws://", "http://").replace("wss://", "https://").rsplit("/", 1)[0] + "/health", timeout=5) as r:
                h = json.loads(r.read())
            out.append({"worker": w.get("id") or w["url"], "active": h.get("active_calls"), "max": h.get("max_calls"), "cap": h.get("cap"), "served": h.get("served"), "refused": h.get("refused")})
        except Exception:  # noqa: BLE001
            out.append({"worker": w.get("id") or w["url"], "active": None})
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gateway", default="http://127.0.0.1:8760")
    ap.add_argument("--origin", default="http://localhost:8000")
    ap.add_argument("--callers", type=int, default=1000, help="quiet callers held open at once")
    ap.add_argument("--active", type=int, default=8, help="real conversations measured before and during")
    ap.add_argument("--ramp-secs", type=float, default=120.0)
    ap.add_argument("--hold-secs", type=float, default=30.0, help="extra time at full size before the measured conversations")
    ap.add_argument("--frame-every", type=float, default=1.0, help="seconds between a quiet caller's 20 ms frames (a microphone sends one every 0.02)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    variants = [(caller_audio(lines, "say:Samantha" if i % 2 == 0 else "say:Daniel"), lines) for i, lines in enumerate(MIX)]
    samples: list[dict] = []
    watcher = asyncio.create_task(watch(samples))
    report: dict = {"callers": a.callers, "active": a.active, "frame_every_s": a.frame_every, "ramp_secs": a.ramp_secs}
    mark = lambda name: report.setdefault("marks", {}).__setitem__(name, time.time())  # noqa: E731
    mark("start")
    print(f"1. baseline: {a.active} real conversations alone", flush=True)
    base = await real_calls(a.gateway, a.origin, a.active, variants, "base")
    report["baseline"] = waits(base)
    print("   ", report["baseline"], flush=True)

    mark("ramp_start")
    print(f"2. ramp: {a.callers} quiet callers over {a.ramp_secs:.0f} s", flush=True)
    quiet = [Quiet(i, a.gateway, a.origin, a.frame_every) for i in range(a.callers)]
    tasks = []
    t0 = time.perf_counter()
    for i, q in enumerate(quiet):
        await asyncio.sleep(max(0.0, i * a.ramp_secs / a.callers - (time.perf_counter() - t0)))
        tasks.append(asyncio.create_task(q.run()))
        if (i + 1) % 100 == 0:
            held = sum(1 for x in quiet if x.status == "open")
            print(f"    started {i + 1}, open now {held}, refused/failed {sum(1 for x in quiet if x.status not in ('new', 'open'))}", flush=True)
    mark("ramp_end")
    await asyncio.sleep(a.hold_secs)
    report["workers_at_peak"] = worker_health(a.gateway)
    print(f"3. hold: {sum(1 for q in quiet if q.status == 'open')} connected; the {a.active} real conversations again", flush=True)
    mark("hold_start")
    busy = await real_calls(a.gateway, a.origin, a.active, variants, "busy")
    mark("hold_end")
    report["with_everyone_connected"] = waits(busy)
    print("   ", report["with_everyone_connected"], flush=True)
    report["still_open_after_hold"] = sum(1 for q in quiet if q.status == "open")

    print("4. end: every quiet caller closes", flush=True)
    for q in quiet:
        q.stop.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    mark("end")
    await asyncio.sleep(10)
    watcher.cancel()
    report["quiet_callers"] = summarise(quiet)
    report["samples"] = samples
    report["workers_after"] = worker_health(a.gateway)
    mem = lambda s, g: round(s["groups"].get(g, (0, 0))[0], 2)  # noqa: E731
    cpu = lambda s, g: round(s["groups"].get(g, (0, 0))[1], 2)  # noqa: E731
    if samples:
        first, peak, last = samples[0], max(samples, key=lambda s: s["machine_mem_used_gb"]), samples[-1]
        groups = sorted({g for s in samples for g in s["groups"]})
        print("\n   group                          memory GB: start / peak / after     cores: start / peak / after")
        for g in groups:
            peak_cores = max(cpu(s, g) for s in samples)
            print(f"   {g:<30} {mem(first, g):>6} / {mem(peak, g):>5} / {mem(last, g):>5}        {cpu(first, g):>5} / {peak_cores:>5} / {cpu(last, g):>5}")
        print(f"   machine: memory used {first['machine_mem_used_gb']:.1f} -> {peak['machine_mem_used_gb']:.1f} GB at the peak; processor busiest {max(s['machine_cpu'] for s in samples):.0f}%")
    print("\nquiet callers:", json.dumps(report["quiet_callers"]))
    print("workers at the peak:", json.dumps(report["workers_at_peak"]))
    if a.out:
        a.out.write_text(json.dumps(report, indent=1))
        print("saved", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
