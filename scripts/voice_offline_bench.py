r"""End-to-end voice timing WITHOUT a microphone: a synthetic caller (a different Piper voice) speaks a scripted conversation into the real
Pipecat pipeline in real time; every stage is timed.

  .venv\Scripts\python.exe scripts\voice_offline_bench.py [--backend openvino_gpu] [--stop-secs 0.35] [--mode conversational|hybrid|template]

Caveat: the caller is synthetic and clean, so STT here is optimistic. Endpointing (VAD) and everything after it is real.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")                                 # Pipecat logs every link at DEBUG

import numpy as np  # noqa: E402
import soxr  # noqa: E402

from dealer_core.app import ROOT, build_app, demo_clock  # noqa: E402
from dealer_core.config import load_config  # noqa: E402
from dealer_core.nlu.chain import BackendChain  # noqa: E402
from dealer_core.voice.offline import run_offline  # noqa: E402
from dealer_core.voice.pipeline import apply_voice_profile, load_engines  # noqa: E402
from dealer_core.voice.tts import PiperSherpa  # noqa: E402

SCRIPT = ["I'm looking for an SUV under one hundred and fifty thousand dirhams", "Toyota please", "Tell me about the second one",
          "Is it GCC spec", "Can you do one hundred and twenty thousand", "How about one hundred and twenty five thousand"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="openvino_gpu")
    ap.add_argument("--stop-secs", type=float, default=None)
    ap.add_argument("--show-text", action="store_true", help="print what the agent said (kept in memory by the harness only; never logged)")
    ap.add_argument("--stt-model", default=None, help="OpenVINO Whisper folder under models/ov (default voice.stt_model)")
    ap.add_argument("--mode", default=None, choices=["conversational", "hybrid", "template"])
    a = ap.parse_args()
    cfg = load_config()
    chain = BackendChain.from_config(cfg, primary=a.backend)
    print("loading models...", flush=True)
    chain.start()
    app = build_app(cfg, chain=chain, clock=demo_clock(cfg), reply_mode=a.mode)
    apply_voice_profile(app, cfg)
    parts = load_engines(cfg, a.stt_model)
    caller = PiperSherpa(ROOT / "models", "amy")                       # a different voice from the agent's, like the earlier benchmarks
    clips = []
    for line in SCRIPT:
        samples = soxr.resample(caller.synthesize(line), caller.sample_rate, 16000)
        clips.append((np.clip(samples, -1, 1) * 32767).astype(np.int16))
    print(f"running {len(clips)} turns in real time (this takes a couple of minutes)...", flush=True)
    res = asyncio.run(run_offline(app, parts, clips, stop_secs=a.stop_secs or cfg.voice.stop_secs))
    print(f"\n{'turn':>4} {'endpoint':>9} {'STT':>6} {'brain':>7} {'TTS 1st':>8} {'TOTAL':>7}   intent / by / model-written")
    for t, b in zip(res.turns, res.brain_log + [{}] * len(res.turns)):
        print(f"{t.index + 1:4d} {t.endpointing_ms:8.0f}ms {t.stt_ms:5.0f}ms {t.brain_ms:6.0f}ms {t.tts_first_ms:7.0f}ms {t.total_ms:6.0f}ms   "
              f"{b.get('intent', '-')} / {b.get('via', '-')} / {b.get('model_written', '-')}  {t.note}"
              + (f"\n       first sentence {b['first_sentence_ms']:.0f} ms; {b['note'][:170]}" if b.get('note') else ""))
    if a.show_text:
        print("\nTRANSCRIPT (YOU = what the speech recognizer heard):")
        for who, text in res.events:
            print(f"  {who:5s}: {text}")
    ok = [t for t in res.turns if t.total_ms]
    if ok:
        tot = [t.total_ms for t in ok]
        print(f"\nend of caller's speech -> first audio: median {st.median(tot):.0f} ms, max {max(tot):.0f} ms over {len(ok)} turns")
        for name in ("endpointing_ms", "stt_ms", "brain_ms", "tts_first_ms"):
            print(f"  {name:15s} median {st.median(getattr(t, name) for t in ok):6.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
