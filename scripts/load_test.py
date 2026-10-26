r"""Several synthetic callers at once through the real WebSocket door (tap-to-talk, the way the page talks).

  .venv\Scripts\python.exe scripts\load_test.py --levels 1 2 3 4 --url ws://127.0.0.1:8766/voice

Start the server first with --max-calls at least as large as the biggest level and on its own port. Each level runs that many identical
conversations at the same moment (the worst case: every caller is in the same phase of the script). Per level: failures, the time from the end
of the caller's speech to the first sound (median, 90th percentile, worst), the share over 2 s, and how long the mic took to close.
Callers' speech is clean synthetic audio, so recognition is optimistic; the timing is real.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
from portal_call import DEFAULT_SCRIPT, caller_audio, one_call_tap  # noqa: E402


MIX = [DEFAULT_SCRIPT,
       ["Hi, I'm looking for a family SUV under one hundred thousand dirhams with low mileage", "What is the difference between the first two you mentioned?", "Which of them has the better service history and is it still under warranty?",
        "Okay, can I come and see the first one on Saturday morning?", "Ten in the morning is fine", "My name is Omar Hassan", "Zero five five nine eight seven six five four three", "Yes that is right"],
       ["Do you have any BMW X5", "Tell me everything about it, the colour, mileage, engine and price", "Is the price negotiable?", "What other cars do you have that are similar to it?", "Thank you that is all for now"],
       ["Good afternoon, I want a Toyota Camry or something similar, not too old", "Show me the cheapest one", "How many kilometres has it done and has it had any accidents?",
        "Can you book me a test drive tomorrow at five in the afternoon", "My name is Layla Ahmed", "My number is zero five zero three three three two two two one", "Yes please confirm it"],
       ["What time are you open today", "Do you have any Toyota", "Compare the Camry and the RAV4", "Tell me about the RAV4", "Does it have a warranty",
        "Has it been in an accident", "How big is the boot", "Can you repeat that", "Thank you"]]


def pct(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else float("nan")


GATEWAY = {"url": ""}


async def staggered(url, origin, clips, i: int, stagger: float):
    await asyncio.sleep(i * stagger)
    if GATEWAY["url"]:                                       # like the page: ask the router which worker takes this call
        import urllib.request

        def assign():
            with urllib.request.urlopen(GATEWAY["url"].rstrip("/") + "/assign", timeout=5) as r:
                return json.loads(r.read())["url"]
        url = await asyncio.to_thread(assign)
    return await one_call_tap(url, clips, None, f"c{i + 1}", origin, None)


async def sample_machine(into: list) -> None:
    import psutil

    psutil.cpu_percent()
    while True:
        await asyncio.sleep(2)
        into.append((psutil.cpu_percent(), psutil.virtual_memory().percent))


async def level(url: str, origin: str, scripts, n: int, stagger: float) -> dict:
    t0 = time.monotonic()
    machine: list = []
    sampler = asyncio.create_task(sample_machine(machine))
    res = await asyncio.gather(*[staggered(url, origin, scripts[i % len(scripts)], i, stagger) for i in range(n)], return_exceptions=True)
    sampler.cancel()
    failed = [r for r in res if isinstance(r, Exception)]
    ok = [r for r in res if not isinstance(r, Exception)]
    first = [t["silence_s"] for r in ok for t in r["turns"] if t["silence_s"] is not None]
    whole = [t["reply_s"] for r in ok for t in r["turns"] if t.get("reply_s")]
    silent = sum(1 for r in ok for t in r["turns"] if t["silence_s"] is None)
    closed = [t["closed_s"] for r in ok for t in r["turns"]]
    taps = sum(1 for r in ok for t in r["turns"] if t["mic_closed_by"] == "tap")
    return {"callers": n, "failed_calls": len(failed), "errors": sorted({type(e).__name__ + ": " + str(e)[:80] for e in failed}), "turns": len(first) + silent,
            "silent_turns": silent, "first_sound_median": round(pct(first, 50), 2), "p90": round(pct(first, 90), 2), "worst": round(max(first), 2) if first else None,
            "over_2s": round(sum(x > 2 for x in first) / len(first), 2) if first else None, "mic_close_median": round(st.median(closed), 2) if closed else None,
            "mic_closed_by_tap": taps, "cpu_mean": round(float(np.mean([c for c, _ in machine])), 0) if machine else None, "cpu_max": max((c for c, _ in machine), default=None), "ram_max": max((m for _, m in machine), default=None), "whole_reply_median": round(pct(whole, 50), 1), "whole_reply_p90": round(pct(whole, 90), 1), "greeting_s": round(st.median([r["greeting_s"] for r in ok]), 1) if ok else None, "wall_s": round(time.monotonic() - t0)}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:8766/voice")
    ap.add_argument("--gateway", default="", help="ask this router for a worker per call (e.g. http://127.0.0.1:8760), like the page does")
    ap.add_argument("--origin", default="http://localhost:8080")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--voice", default="amy")
    ap.add_argument("--mix", action="store_true", help="five different conversations (longer questions, comparisons, a negotiation question) instead of one")
    ap.add_argument("--stagger", type=float, default=0.0, help="seconds between caller starts")
    a = ap.parse_args()
    GATEWAY["url"] = a.gateway
    print("synthesising caller lines...", flush=True)
    scripts = [caller_audio(s, a.voice) for s in (MIX if a.mix else [DEFAULT_SCRIPT])]
    rows = []
    for n in a.levels:
        r = await level(a.url, a.origin, scripts, n, a.stagger)
        rows.append(r)
        print(f"{n} caller(s): {r['turns']} turns, {r['failed_calls']} failed calls, {r['silent_turns']} silent; first sound median {r['first_sound_median']} s, p90 {r['p90']} s, worst {r['worst']} s, "
              f"over 2 s: {r['over_2s']}; whole reply median {r['whole_reply_median']} s, p90 {r['whole_reply_p90']} s; CPU mean {r['cpu_mean']}% max {r['cpu_max']}%, RAM max {r['ram_max']}%; mic closes after {r['mic_close_median']} s; greeting {r['greeting_s']} s of speech; {r['wall_s']} s total {r['errors'] or ''}", flush=True)
        await asyncio.sleep(3)
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "load_test.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
