r"""End-of-turn study: how often does the agent answer while the caller has only paused? And what would a speculative start (begin the work at a short silence, cancel if the
caller resumes) cost and save?

  .venv\Scripts\python.exe scripts\vad_study.py                 # synthetic callers (and recordings/*.wav if present, see scripts/record_samples.py)
  .venv\Scripts\python.exe scripts\vad_study.py --no-pipeline   # only the speculative-start emulation (fast)

Part A (real pipeline, real time): every utterance goes through VAD + Smart Turn + Whisper at VAD silence 0.15 / 0.35 / 0.6. An utterance that produces more than one turn
was cut in the middle. Utterances carry a mid-sentence pause; pauses up to 0.5 s count as "the caller has not finished" (the reference).
Part B (emulation on the same audio, with the real detectors): a speculative start begins the speech recognition and the model at the first VAD silence (0.15 s) when Smart Turn
says the turn is complete, and is cancelled if the caller resumes before the confirmation window (0.35 or 0.6 s of silence) ends. Wasted work = cancelled starts x measured
cost of recognition + tool decision; latency saved = confirmation window - 0.15 s per turn that really ended. This is an emulation with measured component costs, not a pipeline change.
"""
from __future__ import annotations

import argparse
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

LINES = ["I'm looking for|a white SUV", "Can I book|a test drive for tomorrow afternoon", "How much is|the Land Cruiser", "Do you have|anything under one hundred and fifty thousand",
         "Tell me about|the second one", "I want to see|the BMW X5, the grey one", "My name is|Ahmed Khan", "Is it|a GCC spec car", "What about|the kilometres, how many has it done",
         "Hello|I saw a car on your website", "Can you show me|the cheapest one", "I would like|to come and see it today"]
NOMINAL_PAUSES = [0.25, 0.5]             # plus the voice's own ~0.2 s of silence around each piece
REFERENCE = 0.5
STT_S, DECISION_S = 0.20, 0.40           # measured cost of recognition and of the tool decision (live runs, AC)


def synth_clips(tts, speaker_gap: float, lines: list[str]) -> list[np.ndarray]:
    out = []
    for line in lines:
        pieces = [soxr.resample(tts.synthesize(p), tts.sample_rate, 16000) for p in line.split("|")]
        gap = np.zeros(int(16000 * speaker_gap), dtype=np.float32)
        x = np.concatenate([pieces[0], gap, pieces[1]]) if len(pieces) > 1 else pieces[0]
        out.append((np.clip(x, -1, 1) * 32767).astype(np.int16))
    return out


def real_pause_s(x: np.ndarray) -> float:
    """The longest quiet stretch inside the clip (what the detector actually sees), in seconds."""
    loud = np.abs(x.astype(np.float32)) > 330                      # about -40 dBFS
    idx = np.where(loud)[0]
    if len(idx) < 2:
        return 0.0
    gaps = np.diff(idx)
    return float(gaps.max() / 16000) if gaps.max() > 800 else 0.0


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getframerate() != 16000:
            data = (soxr.resample(data.astype(np.float32) / 32768, w.getframerate(), 16000) * 32767).astype(np.int16)
    return data


async def emulate(x: np.ndarray, stop_secs: float, confirm: float, analyzer) -> dict:
    """Walk the clip with the real Silero VAD (stop_secs) and Smart Turn. A 'start' = VAD says stopped AND Smart Turn says complete."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams, VADState

    vad = SileroVADAnalyzer(sample_rate=16000, params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=stop_secs))
    vad.set_sample_rate(16000)
    analyzer.clear()
    frame = 320
    t, speaking, starts, resume_after = 0.0, False, [], None
    pad = np.concatenate([x, np.zeros(16000 * 2, dtype=np.int16)])
    for k in range(0, len(pad) - frame, frame):
        chunk = pad[k:k + frame].tobytes()
        state = await vad.analyze_audio(chunk)
        is_speech = state in (VADState.STARTING, VADState.SPEAKING)
        analyzer.append_audio(chunk, is_speech)
        now = k / 16000
        if state is VADState.SPEAKING and not speaking:
            speaking = True
            if starts and starts[-1]["resumed_at"] is None:
                starts[-1]["resumed_at"] = now                       # the caller carried on after a start
        elif state is VADState.QUIET and speaking:
            speaking = False
            verdict, _ = await analyzer.analyze_end_of_turn()
            if verdict.name == "COMPLETE":
                starts.append({"at": now, "resumed_at": None})
    wasted = [s for s in starts if s["resumed_at"] is not None and s["resumed_at"] - s["at"] < confirm]
    committed = [s for s in starts if s not in wasted]
    return {"starts": len(starts), "cancelled": len(wasted), "committed": len(committed)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pipeline", action="store_true", help="skip part A (the slow real-time pipeline run)")
    ap.add_argument("--recordings", default=str(ROOT / "recordings"))
    a = ap.parse_args()
    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    from dealer_core.config import load_config
    from dealer_core.voice.tts import PiperSherpa

    cfg = load_config()
    caller = PiperSherpa(ROOT / cfg.app.model_dir, "joe")
    utter: list[dict] = []
    for gap in NOMINAL_PAUSES:
        for line, clip in zip(LINES, synth_clips(caller, gap, LINES)):
            utter.append({"id": f"synthetic gap {gap}: {line.replace('|', ' / ')}", "audio": clip, "source": "synthetic", "pause": real_pause_s(clip)})
    for line, clip in zip(LINES, synth_clips(caller, 0.0, [l.replace("|", " ") for l in LINES])):
        utter.append({"id": f"synthetic no pause: {line.replace('|', ' ')}", "audio": clip, "source": "synthetic-control", "pause": 0.0})
    rec = sorted(Path(a.recordings).glob("*.wav")) if Path(a.recordings).exists() else []
    for p in rec:
        x = read_wav(p)
        utter.append({"id": f"recording {p.name}", "audio": x, "source": "recording", "pause": real_pause_s(x)})
    print(f"{len(utter)} utterances: {sum(u['source'] == 'synthetic' for u in utter)} synthetic with a pause, "
          f"{sum(u['source'] == 'synthetic-control' for u in utter)} synthetic controls, {len(rec)} recordings of your voice")

    results = {"settings": []}
    if not a.no_pipeline:
        from dealer_core.app import build_app
        from dealer_core.dialogue.states import Intent
        from dealer_core.nlu.backend import IntentResult
        from dealer_core.voice.offline import run_offline
        from dealer_core.voice.pipeline import load_engines

        class Dummy:
            name = "dummy"

            def classify(self, state, car, text, last_agent=""):
                return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

        app = build_app(load_config(overlay=ROOT / "config" / "portal.yaml"), inventory="portal", portal_data=Path(r"C:\dev\bmw-portal\site\data.js"),
                        classifier=Dummy(), reply_mode="template")
        parts = load_engines(cfg)
        for stop in (0.15, 0.35, 0.6):
            t0 = time.time()
            res = asyncio.run(run_offline(app, parts, [u["audio"] for u in utter], stop_secs=stop, smart_turn=True, smart_fallback_secs=cfg.voice.smart_fallback_secs,
                                          timeout_s=3000))
            heard = [t for who, t in res.events if who == "YOU"]
            row = {"vad_silence": stop, "utterances": len(utter), "turns_answered": len(heard), "extra_turns": max(0, len(heard) - len(utter)),
                   "endpoint_ms_median": float(np.median([t.endpointing_ms for t in res.turns if t.endpointing_ms])) if res.turns else None}
            results["settings"].append(row)
            print(f"A. VAD silence {stop:.2f} s + Smart Turn: {len(utter)} utterances -> {len(heard)} turns ({row['extra_turns']} cut mid-sentence), median wait "
                  f"{row['endpoint_ms_median']:.0f} ms [{time.time() - t0:.0f} s]", flush=True)

    analyzer = LocalSmartTurnAnalyzerV3(sample_rate=16000, params=SmartTurnParams(stop_secs=cfg.voice.smart_fallback_secs))
    analyzer.set_sample_rate(16000)
    print("\nB. Speculative start (emulation, same audio, real detectors):")
    results["speculative"] = []
    for confirm in (0.35, 0.6):
        tot = {"starts": 0, "cancelled": 0, "committed": 0}
        for u in utter:
            r = asyncio.run(emulate(u["audio"], 0.15, confirm, analyzer))
            for k in tot:
                tot[k] += r[k]
        n = len(utter)
        saved = tot["committed"] * (confirm - 0.15)
        wasted_work = tot["cancelled"] * (STT_S + DECISION_S)
        row = {"confirm_window_s": confirm, **tot, "wasted_work_s": round(wasted_work, 1), "latency_saved_per_turn_s": round(confirm - 0.15, 2),
               "wasted_fraction_of_starts": round(tot["cancelled"] / max(tot["starts"], 1), 2)}
        results["speculative"].append(row)
        print(f"   confirm window {confirm:.2f} s: {tot['starts']} speculative starts over {n} utterances, {tot['cancelled']} cancelled ({100 * row['wasted_fraction_of_starts']:.0f}%) = "
              f"{wasted_work:.1f} s of wasted recognition+decision work; each committed turn answers {confirm - 0.15:.2f} s sooner than waiting for a {confirm:.2f} s silence "
              f"({tot['committed']} turns, {saved:.1f} s saved in total)")
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "vad_study.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
