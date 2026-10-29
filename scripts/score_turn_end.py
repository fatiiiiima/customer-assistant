r"""Tap mode: how long should the detector's silence be? Recorded turns replayed at several settings: sentences split in the middle, whether the join saves them,
and how long the caller waits after really finishing.

  uv run python scripts/score_turn_end.py                       # every recorded call in logs/audio_dump, at 0.6, 0.35 and 0.2 s
  uv run python scripts/score_turn_end.py --stop 0.6 0.45 0.35 --verbose

Each turn of a recorded call (from a tap to the end of the turn: one sentence as the caller said it, the live system having waited 0.6 s) is played through the REAL
Silero detector and the REAL SmartEndpointer (Smart Turn v3) at each silence, as `score_overlap.py` does. A turn end before the caller's last loud frame splits the
sentence. Since 27 Oct a split is not lost: the microphone stays open until the reply starts to play (at most `voice.tap_grace_secs`), and a caller who carries on is
joined to what they said before (`micgate.ContinuationGuard`). So a split is SAVED when the caller is heard again (VAD start, 0.1 s) before the reply plays. When the
reply plays depends on the reply: a sentence the voice has ready (a booking question) about 0.3 s after the turn end, one the model writes about 1 s or more. Both are
reported. The wait is from the caller's last loud frame to the final turn end (Whisper and the reply come after it).
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from hearing_check import load_turns  # noqa: E402
from score_overlap import replay  # noqa: E402

REPLY_AT = {"ready reply (0.3 s)": 0.3, "model reply (1.0 s)": 1.0}
VAD_START_S = 0.1


def loud_frames(clip: np.ndarray) -> np.ndarray:
    """Times (s) of the clip's 20 ms frames that are well above its background."""
    n = len(clip) // 320
    level = np.sqrt(np.mean(clip[:n * 320].reshape(n, 320) ** 2, axis=1)) if n else np.zeros(0)
    if not n:
        return np.zeros(0)
    loud = max(0.01, 4 * float(np.percentile(level, 20)))
    return np.where(level > loud)[0] * 0.02


def score(stops: list[float], loud: np.ndarray, grace_s: float) -> dict:
    """A turn's splits and wait, from the replay's turn ends and the clip's loud frames."""
    if not len(loud) or not stops:
        return {}
    spoke_end = float(loud[-1]) + 0.02
    splits = [t for t in stops if t < spoke_end - 0.05]
    saved = {}
    for label, reply_at in REPLY_AT.items():
        ok = 0
        for t in splits:
            again = loud[loud > t]
            ok += bool(len(again)) and (float(again[0]) - t + VAD_START_S) < min(reply_at, grace_s)
        saved[label] = ok
    final = [t for t in stops if t >= spoke_end - 0.05]
    return {"splits": len(splits), "saved": saved, "wait": (final[0] - spoke_end) if final else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="*", help="recordings in logs/audio_dump without .pcm (default: every one with caller audio)")
    ap.add_argument("--folder", type=Path, default=ROOT / "logs" / "audio_dump")
    ap.add_argument("--stop", type=float, nargs="+", default=[0.6, 0.35, 0.2], help="the detector's silences to compare (seconds)")
    ap.add_argument("--pace", type=float, default=0.05, help="1 = real time; the default plays 20x faster (the detector and Smart Turn count samples, not seconds)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    from dealer_core.config import load_config

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    analyzer = LocalSmartTurnAnalyzerV3(sample_rate=16000, params=SmartTurnParams(stop_secs=cfg.voice.smart_fallback_secs))
    analyzer.set_sample_rate(16000)
    stems = a.stems or sorted(p.name[:-len(".pcm")] for p in a.folder.glob("*_in.pcm") if p.stat().st_size > 0)
    clips = [(stem, i, c) for stem in stems for i, c in enumerate(load_turns(stem, a.folder, None))]
    print(f"{len(clips)} recorded turns from {len(stems)} calls; Smart Turn fallback {cfg.voice.smart_fallback_secs} s, join window {cfg.voice.tap_grace_secs} s\n")
    results = {}
    for stop in a.stop:
        rows = []
        for stem, i, clip in clips:
            out = asyncio.run(replay(clip, 0.0, cfg, analyzer, stop, a.pace))
            r = score(out["stops"], loud_frames(clip), cfg.voice.tap_grace_secs)
            if r:
                rows.append(r)
                if a.verbose and r["splits"]:
                    print(f"  {stop:.2f} s  {stem} turn {i}: split {r['splits']}x, saved {r['saved']}")
        waits = [r["wait"] for r in rows if r["wait"] is not None]
        results[stop] = {"turns": len(rows), "split": sum(r["splits"] for r in rows), "turns_split": sum(1 for r in rows if r["splits"]),
                         **{f"saved, {k}": sum(r["saved"][k] for r in rows) for k in REPLY_AT},
                         "wait_median": statistics.median(waits) if waits else None, "wait_p90": sorted(waits)[int(0.9 * (len(waits) - 1))] if waits else None}
        x = results[stop]
        print(f"silence {stop:.2f} s: {x['turns']} turns, {x['split']} splits in {x['turns_split']} turns; saved by the join: "
              + ", ".join(f"{x[f'saved, {k}']} ({k})" for k in REPLY_AT) + f"; wait after the caller finished: median {x['wait_median']:.2f} s, p90 {x['wait_p90']:.2f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
