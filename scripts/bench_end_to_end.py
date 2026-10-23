r"""REAL end-to-end silence before the first sound: a synthetic caller speaks 5 scripted chats (search, price question, negotiation, booking, follow-up) into the real
Pipecat pipeline (VAD + Smart Turn + Whisper + engine + Piper) in real time, and every turn is timed from the end of the caller's speech to the first audio sample.

  .venv\Scripts\python.exe scripts\bench_end_to_end.py --engine state_machine
  .venv\Scripts\python.exe scripts\bench_end_to_end.py --engine agent
  .venv\Scripts\python.exe scripts\bench_end_to_end.py --engine agent --prefix

Run it on AC power with the Windows power mode on "Best performance"; it refuses to run on battery. The result is also written to logs/e2e_<label>.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import soxr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHATS = {
    "search": ["I'm looking for an SUV", "under one fifty thousand", "a white one", "tell me about the second one"],
    "price": ["BMW X5", "how much is it", "is that a good price", "how many kilometres has it done"],
    "negotiation": ["Toyota Land Cruiser", "how much is it", "can you do one hundred and fifty thousand", "what is your best price"],
    "booking": ["Mercedes GLE", "the GLE please", "I'd like a test drive", "tomorrow at three in the afternoon", "my name is Ahmed Khan",
                "my number is zero five zero one two three four five six seven", "yes that is right"],
    "follow-up": ["Toyota", "any budget", "tell me about the second one", "how much is it"],
}


def on_ac_power() -> bool:
    try:
        import psutil

        b = psutil.sensors_battery()
        return b is None or bool(b.power_plugged)
    except Exception:
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["state_machine", "agent"], default="state_machine")
    ap.add_argument("--prefix", action="store_true")
    ap.add_argument("--backend", default="openvino_gpu")
    ap.add_argument("--vad-silence", type=float, default=None, help="VAD silence; default voice.smart_stop_secs")
    ap.add_argument("--allow-battery", action="store_true", help="run anyway (the numbers are then DISCARDED, labelled as such)")
    ap.add_argument("--chats", nargs="*", default=list(CHATS))
    a = ap.parse_args()
    ac = on_ac_power()
    if not ac and not a.allow_battery:
        print("This laptop is on battery: timings would be meaningless. Plug in (and set Best performance), or pass --allow-battery to run anyway (results labelled DISCARDED).")
        return 2

    from dealer_core.app import build_app
    from dealer_core.cli.chat import start
    from dealer_core.config import load_config
    from dealer_core.voice.offline import run_offline
    from dealer_core.voice.pipeline import apply_voice_profile, load_engines
    from dealer_core.voice.tts import PiperSherpa

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=Path(r"C:\dev\bmw-portal\site\data.js"), backend=a.backend,
                    engine="agent" if a.engine == "agent" else None, agent_prefix=a.prefix if a.engine == "agent" else None)
    start(app, quiet=True)
    apply_voice_profile(app, cfg)
    parts = load_engines(cfg)
    caller = PiperSherpa(ROOT / "models", "joe")
    label = a.engine + ("+prefix" if a.prefix else "") + ("" if ac else "_ON_BATTERY_DISCARDED")
    stop = a.vad_silence or cfg.voice.smart_stop_secs
    rows = []
    for name in a.chats:
        lines = CHATS[name]
        clips = [(np.clip(soxr.resample(caller.synthesize(t), caller.sample_rate, 16000), -1, 1) * 32767).astype(np.int16) for t in lines]
        t0 = time.time()
        res = asyncio.run(run_offline(app, parts, clips, stop_secs=stop, smart_turn=cfg.voice.smart_turn, smart_fallback_secs=cfg.voice.smart_fallback_secs,
                                      speak_opening=False, timeout_s=400))
        heard = [t for who, t in res.events if who == "YOU"]
        said: list[list[str]] = []
        for who, t in res.events:
            if who == "YOU":
                said.append([])
            elif said:
                said[-1].append(t)
        for i, (line, tt) in enumerate(zip(lines, res.turns)):
            rows.append({"chat": name, "customer": line, "heard": heard[i] if i < len(heard) else None, "silence_s": tt.total_ms / 1000 if tt.total_ms else None,
                         "endpoint_ms": tt.endpointing_ms, "stt_ms": tt.stt_ms, "brain_ms": tt.brain_ms, "tts_ms": tt.tts_first_ms,
                         "reply": " ".join(said[i]) if i < len(said) else ""})
        print(f"{name}: {len(lines)} turns in {time.time() - t0:.0f} s", flush=True)
    ok = [r["silence_s"] for r in rows if r["silence_s"]]
    ok_sorted = sorted(ok)
    print(f"\n=== {label}  (AC power: {ac}; engine {a.engine}{' + prefix' if a.prefix else ''}; VAD silence {stop} s; Smart Turn {cfg.voice.smart_turn})")
    print(f"{'chat':12s} {'customer':50s} {'silence':>8s}  endpoint/STT/brain/TTS (ms)")
    for r in rows:
        print(f"{r['chat']:12s} {r['customer'][:50]:50s} {('%.2f s' % r['silence_s']) if r['silence_s'] else 'NO AUDIO':>8s}  "
              f"{r['endpoint_ms']:.0f}/{r['stt_ms']:.0f}/{r['brain_ms']:.0f}/{r['tts_ms']:.0f}")
    if ok:
        print(f"\nsilence before the first sound over {len(ok)} turns: median {st.median(ok):.2f} s, mean {st.mean(ok):.2f} s, p90 {ok_sorted[int(0.9 * (len(ok_sorted) - 1))]:.2f} s, "
              f"worst {max(ok):.2f} s; over 2 s: {sum(x > 2 for x in ok)}; turns with no audio: {len(rows) - len(ok)}")
    Path(ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / f"e2e_{label}.json").write_text(json.dumps({"label": label, "ac_power": ac, "rows": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
