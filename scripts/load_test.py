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


def pct(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else float("nan")


async def level(url: str, origin: str, clips, n: int) -> dict:
    t0 = time.monotonic()
    res = await asyncio.gather(*[one_call_tap(url, clips, None, f"c{i + 1}", origin, None) for i in range(n)], return_exceptions=True)
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
            "mic_closed_by_tap": taps, "whole_reply_median": round(pct(whole, 50), 1), "whole_reply_p90": round(pct(whole, 90), 1), "greeting_s": round(st.median([r["greeting_s"] for r in ok]), 1) if ok else None, "wall_s": round(time.monotonic() - t0)}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:8766/voice")
    ap.add_argument("--origin", default="http://localhost:8080")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--voice", default="amy")
    a = ap.parse_args()
    print("synthesising caller lines...", flush=True)
    clips = caller_audio(DEFAULT_SCRIPT, a.voice)
    rows = []
    for n in a.levels:
        r = await level(a.url, a.origin, clips, n)
        rows.append(r)
        print(f"{n} caller(s): {r['turns']} turns, {r['failed_calls']} failed calls, {r['silent_turns']} silent; first sound median {r['first_sound_median']} s, p90 {r['p90']} s, worst {r['worst']} s, "
              f"over 2 s: {r['over_2s']}; whole reply median {r['whole_reply_median']} s, p90 {r['whole_reply_p90']} s; mic closes after {r['mic_close_median']} s; greeting {r['greeting_s']} s of speech; {r['wall_s']} s total {r['errors'] or ''}", flush=True)
        await asyncio.sleep(3)
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "load_test.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
