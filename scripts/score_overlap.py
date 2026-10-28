r"""Hearing and overlap on recorded turns: does starting recognition earlier (voice.early_start_secs) hear the same words, and how much sooner?

  uv run python scripts/score_overlap.py                              # every recording in logs/audio_dump with a words log, base model
  uv run python scripts/score_overlap.py 20261027_215011_9456_in       # one call (it has reference lines: scored against what was really said)
  uv run python scripts/score_overlap.py --model whisper-small.en-int8-ov --early 0.2 0.3

Each turn of a recorded call (the stretch between a tap and the end of the turn, as `hearing_check.py` cuts it) is played through the REAL Silero detector (tap-mode silence 0.6 s),
the REAL SmartEndpointer and the REAL recogniser service, with the recogniser's engine replaced by one that only records the audio it is given. Twice: the normal path, and
with the early start on. So the audio recognised below is exactly what the live pipeline would have recognised, not a slice cut by hand. Then both are recognised with the real
Whisper and compared.

  normal        the segment the recogniser gets after the turn is confirmed over
  early         the slice taken after `early` seconds of quiet inside the turn, when the turn then ended (a "used" start); a start the caller then spoke over is "wasted"
  saved         audio-clock seconds from the early start to the confirmed end of turn: how much sooner recognition could begin
  agree         word error between the early reading and the normal reading of the same turn (no reference needed: 0% = the same words)
  vs ref        both readings against the reference lines, when there are any (only the 27 Oct call has them; the others are not scored for truth)
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from hearing_check import lines, load_turns, wer  # noqa: E402

FRAME = 320                                               # 20 ms of 16 kHz
TAIL_S = 1.6                                              # silence after the recorded turn, so the detector and Smart Turn can finish


def frames_of(clip: np.ndarray) -> list[bytes]:
    x = np.concatenate([(clip * 32768).clip(-32768, 32767).astype("<i2"), np.zeros(int(TAIL_S * 16000), dtype="<i2")])
    x = np.pad(x, (0, -len(x) % FRAME))
    return [x[i:i + FRAME].tobytes() for i in range(0, len(x), FRAME)]


async def replay(clip: np.ndarray, early_start_secs: float, cfg, analyzer, stop_secs: float, pace: float) -> dict:
    """One turn through the real detector, endpointer and recogniser service. Returns what the recogniser would have been given."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.frames.frames import EndFrame, Frame, InputAudioRawFrame, VADUserStoppedSpeakingFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.workers.runner import WorkerRunner

    from dealer_core.voice.endpoint import SmartEndpointer
    from dealer_core.voice.services import OpenVinoWhisperSTTService
    from dealer_core.voice.speculate import Speculation

    v = cfg.voice
    out = {"segments": [], "starts": [], "stops": [], "used": 0, "dropped": 0}

    class Engine:
        name = "capture"

        def transcribe(self, pcm: bytes, prompt=None):
            out["segments"].append((endpointer.t, pcm))
            return "", 0.0

    class Source(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)

    class Sink(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)

    class TimedEndpointer(SmartEndpointer):
        """The real endpointer, with a clock of its own: the seconds of audio IT has been given. (Run faster than real time the pipeline lags the source, so the source's
        count would put every event later than it was.)"""
        t = 0.0

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            if isinstance(frame, InputAudioRawFrame):
                self.t += len(frame.audio) / 32000
            await super().process_frame(frame, direction)

        async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
            if isinstance(frame, VADUserStoppedSpeakingFrame):
                out["stops"].append(self.t)
            await super().push_frame(frame, direction)

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(sample_rate=16000, params=VADParams(
        confidence=v.vad_confidence, start_secs=v.vad_start_secs, stop_secs=stop_secs, min_volume=v.vad_min_volume)))
    clear = getattr(analyzer, "clear", None)
    if clear:
        clear()
    endpointer = TimedEndpointer(16000, v.smart_fallback_secs, analyzer=analyzer, short_answer_secs=v.smart_short_answer_secs, early_start_secs=early_start_secs)

    def record_start(pcm: bytes) -> str:
        out["starts"].append((endpointer.t, pcm))
        return "x"                                         # any text: the real words are read afterwards from the audio

    spec = Speculation(record_start, None, lambda: 0)
    endpointer.speculation = spec
    stt = OpenVinoWhisperSTTService(engine=Engine(), min_rms=v.min_rms)
    stt.speculation = spec
    source = Source()
    pipeline = Pipeline([source, vad, endpointer, stt, Sink()])
    worker = PipelineWorker(pipeline, params=PipelineParams(audio_in_sample_rate=16000), idle_timeout_secs=None)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    async def play():
        await asyncio.sleep(0.3)
        for chunk in frames_of(clip):
            await source.push_frame(InputAudioRawFrame(audio=chunk, sample_rate=16000, num_channels=1))
            await asyncio.sleep(0.02 * pace)
        await asyncio.sleep(0.5)
        await worker.queue_frame(EndFrame())

    player = asyncio.create_task(play())
    try:
        await asyncio.wait_for(runner.run(), timeout=120)
    finally:
        player.cancel()
    out["used"], out["dropped"], out["started"] = spec.used, spec.dropped, spec.started
    return out


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="*", help="recordings in logs/audio_dump without .pcm (default: every one that has a words log)")
    ap.add_argument("--folder", type=Path, default=ROOT / "logs" / "audio_dump")
    ap.add_argument("--model", default=None, help="a recogniser folder under models/ov (default voice.stt_model)")
    ap.add_argument("--early", type=float, nargs="+", default=[0.2], help="seconds of quiet inside a turn after which recognition starts (voice.early_start_secs)")
    ap.add_argument("--pseudo-ref", default=None, help="a larger recogniser (folder under models/ov, e.g. whisper-small.en-int8-ov) whose reading of the NORMAL segment stands in for the truth on turns "
                    "that have no reference lines. PSEUDO: it is a recogniser's reading, not what was said; it tells which of two readings is nearer a better listener")
    ap.add_argument("--pad", type=float, default=0.0, help="append this many seconds of digital silence to an early slice before it is recognised (the normal segment ends in the detector's 0.6 s of silence; a slice "
                    "cut at the first 0.2 s of quiet stops abruptly after the last word)")
    ap.add_argument("--stop-secs", type=float, default=None, help="the detector's silence (default voice.tap_stop_secs, the website's tap mode)")
    ap.add_argument("--pace", type=float, default=0.05, help="1 = real time; the default plays 20x faster than real time (the detector counts samples, not seconds)")
    ap.add_argument("--verbose", action="store_true", help="one line per turn: when the early start began, when the turn was confirmed over, what each reading was")
    ap.add_argument("--limit", type=int, default=None, help="at most this many turns per recording")
    a = ap.parse_args()

    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    from dealer_core.app import build_app
    from dealer_core.config import load_config
    from dealer_core.dialogue.states import State
    from dealer_core.voice.prompts import hotword_groups, stt_prompt
    from dealer_core.voice.services import clean_transcript
    from dealer_core.voice.stt import WhisperOpenVino, hotword_terms

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    stop_secs = a.stop_secs if a.stop_secs is not None else cfg.voice.tap_stop_secs
    hotwords = hotword_terms(ROOT / cfg.stt.hotwords_file)
    groups = hotword_groups(ROOT / cfg.stt.hotwords_file)
    site = ROOT.parent / "bmw-portal" / "site" / "data.js"
    app = build_app(cfg, inventory="portal" if site.exists() else "demo", portal_data=site if site.exists() else None, reply_mode="template",
                    classifier=type("Stub", (), {"name": "stub", "classify": lambda *x, **k: None})())
    vocab = app.inventory.vocabulary()
    model = a.model or cfg.voice.stt_model
    engine = WhisperOpenVino(ROOT / "models" / "ov" / model, ROOT / "models" / "ov_cache", "CPU", hotwords, precision=cfg.voice.stt_precision)
    engine.warm_up()
    analyzer = LocalSmartTurnAnalyzerV3(sample_rate=16000, params=SmartTurnParams(stop_secs=cfg.voice.smart_fallback_secs))
    analyzer.set_sample_rate(16000)

    stems = a.stems or sorted({p.name[:-len("_in.txt")] + "_in" for p in a.folder.glob("*_in.txt")} | {p.name[:-len(".ref.txt")] for p in a.folder.glob("*.ref.txt")})
    judge = None
    if a.pseudo_ref:
        judge = WhisperOpenVino(ROOT / "models" / "ov" / a.pseudo_ref, ROOT / "models" / "ov_cache", "CPU", hotwords, precision=cfg.voice.stt_precision)
        judge.warm_up()
    rows: list[dict] = []
    for stem in stems:
        if not (a.folder / f"{stem}.pcm").exists():
            continue
        ref, agent = lines(a.folder / f"{stem}.ref.txt"), lines(a.folder / f"{stem}.agent.txt")
        clips = load_turns(stem, a.folder, a.limit)
        for i, clip in enumerate(clips):
            hint = stt_prompt(State.ANSWERING, agent[i] if i < len(agent) else "", groups, vocab)

            def read(pcm: bytes) -> tuple[str, float]:
                t0 = time.perf_counter()
                text = clean_transcript(engine.transcribe_float(pcm_to_float(pcm), hint))
                return text, (time.perf_counter() - t0) * 1000

            normal = asyncio.run(replay(clip, 0.0, cfg, analyzer, stop_secs, a.pace))
            if not normal["segments"]:
                rows.append({"stem": stem, "turn": i, "skipped": "the detector found no speech (or it was below min_rms)"})
                continue
            seg_t, seg = normal["segments"][-1]
            text_n, ms_n = read(seg)
            pseudo = clean_transcript(judge.transcribe_float(pcm_to_float(seg), hint)) if judge is not None else None
            end_t = (normal["stops"] or [seg_t])[-1]
            for early in a.early:
                e = asyncio.run(replay(clip, early, cfg, analyzer, stop_secs, a.pace))
                row = {"stem": stem, "turn": i, "early": early, "normal": text_n, "normal_ms": ms_n, "seg_s": len(seg) / 32000, "starts": e["started"],
                       "used": e["used"], "wasted": e["started"] - e["used"], "ref": ref[i] if i < len(ref) else None, "pseudo": pseudo}
                if e["used"] and e["starts"]:
                    st, pcm = e["starts"][-1]
                    row["early_text"], row["early_ms"] = read(pcm + bytes(int(a.pad * 32000) // 2 * 2))
                    row["slice_s"] = len(pcm) / 32000
                    row["saved_s"] = max(0.0, end_t - st)
                    row["start_t"], row["end_t"] = st, end_t
                rows.append(row)

    for early in a.early:
        mine = [r for r in rows if r.get("early") == early]
        used = [r for r in mine if "early_text" in r]
        print(f"\n== early start after {early:.2f} s of quiet" + (f", slice padded with {a.pad:.2f} s of silence" if a.pad else "") + f", detector silence {stop_secs} s, recogniser {model} ({len(mine)} turns in {len(stems)} recordings)")
        print(f"turns with a usable early start: {len(used)} of {len(mine)}; recognition runs started: {sum(r['starts'] for r in mine)} (wasted: {sum(r['wasted'] for r in mine)}); "
              f"extra runs per turn: {sum(r['wasted'] for r in mine) / max(1, len(mine)):.2f}")
        if used:
            ag = [wer(r["normal"], r["early_text"]) for r in used]
            exact = sum(e == 0 for e, _ in ag)
            print(f"agreement with the normal reading: {exact} of {len(used)} turns identical; word difference {100 * sum(e for e, _ in ag) / max(1, sum(n for _, n in ag)):.1f}% "
                  f"({sum(e for e, _ in ag)} of {sum(n for _, n in ag)} words)")
            print(f"recognition could start {statistics.median(r['saved_s'] for r in used):.2f} s sooner (median; mean {statistics.fmean(r['saved_s'] for r in used):.2f}); "
                  f"slice {statistics.median(r['slice_s'] for r in used):.1f} s against segment {statistics.median(r['seg_s'] for r in used):.1f} s (median); "
                  f"recognition {statistics.median(r['early_ms'] for r in used):.0f} ms early, {statistics.median(r['normal_ms'] for r in used):.0f} ms normal")
            gains = [min(r["saved_s"], r["normal_ms"] / 1000) for r in used]           # the wait removed: the recognition run, up to the time the early start gained (if the CPU is free for it)
            print(f"the reply could begin {statistics.median(gains):.2f} s sooner (median over the turns with a usable start; {len(used) / max(1, len(mine)):.0%} of turns): "
                  f"{sum(g >= r['normal_ms'] / 1000 - 0.01 for g, r in zip(gains, used))} of {len(used)} get the whole recognition run for free")
        scored = [r for r in mine if r["ref"]]
        if scored:
            en = [wer(r["ref"], r["normal"]) for r in scored]
            ee = [wer(r["ref"], r.get("early_text", r["normal"])) for r in scored]
            print(f"against what was said ({len(scored)} turns with reference lines; an unused early start counts as the normal reading): "
                  f"normal {100 * sum(e for e, _ in en) / max(1, sum(n for _, n in en)):.1f}% words wrong, early {100 * sum(e for e, _ in ee) / max(1, sum(n for _, n in ee)):.1f}%")
        judged = [r for r in used if r["pseudo"] is not None and not r["ref"]]
        if judged:
            dn = [wer(r["pseudo"], r["normal"]) for r in judged]
            de = [wer(r["pseudo"], r["early_text"]) for r in judged]
            nearer = sum(e < n for (e, _), (n, _) in zip(de, dn))
            further = sum(e > n for (e, _), (n, _) in zip(de, dn))
            print(f"against the PSEUDO-reference ({a.pseudo_ref}, {len(judged)} turns with a usable start and no reference lines): normal {100 * sum(e for e, _ in dn) / max(1, sum(n for _, n in dn)):.1f}% "
                  f"of words differ, early {100 * sum(e for e, _ in de) / max(1, sum(n for _, n in de)):.1f}%; the early reading is nearer on {nearer} turns, further on {further}, equal on {len(judged) - nearer - further}")
        for r in (used if a.verbose else []):
            print(f"  [{r['stem'][:15]} turn {r['turn']}] early start at {r['start_t']:.2f} s, turn confirmed over at {r['end_t']:.2f} s (saved {r['saved_s']:.2f}); starts {r['starts']}, "
                  f"slice {r['slice_s']:.2f} s, segment {r['seg_s']:.2f} s: {r['normal']!r}")
        for r in used:
            if wer(r["normal"], r["early_text"])[0]:
                print(f"  differs [{r['stem'][:15]} turn {r['turn']}]  normal: {r['normal']!r}\n  {'':35s}early : {r['early_text']!r}")
    skipped = [r for r in rows if "skipped" in r]
    if skipped:
        print(f"\n{len(skipped)} turns skipped (no speech found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
