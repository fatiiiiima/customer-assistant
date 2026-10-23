r"""Your real recordings through the real pipeline: what the recognizer heard, whether a sentence was cut in the middle at your natural pause, and how long the end-of-turn wait was.

  .venv\Scripts\python.exe scripts\study_recordings.py                # recordings/*.wav (made with scripts/record_samples.py)

For each setting (VAD silence, Smart Turn fallback, words rule) every recording is played in real time; a recording that produces more than one turn was cut at your pause.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import soxr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SETTINGS = [("VAD 0.20 s, fallback 1.0 s (old)", 0.2, 1.0, False), ("VAD 0.20 s, fallback 0.6 s (current)", 0.2, 0.6, False),
            ("VAD 0.20 s, fallback 0.6 s + words rule", 0.2, 0.6, True), ("VAD 0.35 s, fallback 0.6 s", 0.35, 0.6, False), ("VAD 0.60 s, fallback 0.6 s", 0.6, 0.6, False)]


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getframerate() != 16000:
            x = (soxr.resample(x.astype(np.float32) / 32768, w.getframerate(), 16000) * 32767).astype(np.int16)
    return x


def pause_s(x: np.ndarray) -> float:
    loud = np.where(np.abs(x.astype(np.float32)) > 400)[0]
    if len(loud) < 2:
        return 0.0
    g = np.diff(loud)
    return float(g.max() / 16000) if g.max() > 1600 else 0.0


def main() -> int:
    from dealer_core.app import build_app
    from dealer_core.config import load_config
    from dealer_core.dialogue.states import Intent
    from dealer_core.nlu.backend import IntentResult
    from dealer_core.voice.offline import run_offline
    from dealer_core.voice.pipeline import load_engines

    class Dummy:
        name = "dummy"

        def classify(self, state, car, text, last_agent=""):
            return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

    meta = json.loads((ROOT / "recordings" / "index.json").read_text(encoding="utf-8"))
    names = sorted(meta)
    clips = [read_wav(ROOT / "recordings" / n) for n in names]
    pauses = [pause_s(c) for c in clips]
    print(f"{len(clips)} recordings; the longest quiet stretch inside them: median {np.median(pauses):.2f} s, range {min(pauses):.2f}-{max(pauses):.2f} s; "
          f"{sum(p > 0.5 for p in pauses)} of them pause longer than the 0.5 s reference", flush=True)
    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=Path(r"C:\dev\bmw-portal\site\data.js"), classifier=Dummy(), reply_mode="template")
    parts = load_engines(cfg)
    out = {"pauses": dict(zip(names, pauses)), "settings": []}
    for label, stop, fallback, peek in SETTINGS:
        t0 = time.time()
        res = asyncio.run(run_offline(app, parts, clips, stop_secs=stop, smart_turn=True, smart_fallback_secs=fallback, smart_short_answer_secs=0.0, smart_peek=peek,
                                      speak_opening=True, timeout_s=3000))
        heard = [t for who, t in res.events if who == "YOU"]
        waits = [t.endpointing_ms for t in res.turns if t.endpointing_ms]
        row = {"setting": label, "recordings": len(clips), "turns": len(heard), "cut": max(0, len(heard) - len(clips)),
               "wait_median_ms": float(np.median(waits)) if waits else None, "wait_p90_ms": float(np.percentile(waits, 90)) if waits else None, "heard": heard}
        out["settings"].append(row)
        print(f"{label:44s}: {len(clips)} recordings -> {len(heard)} turns ({row['cut']} cut at a pause), wait median {row['wait_median_ms']:.0f} ms, p90 {row['wait_p90_ms']:.0f} ms [{time.time() - t0:.0f} s]", flush=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "study_recordings.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
