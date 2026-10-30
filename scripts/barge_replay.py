"""Replay recorded hands-free calls (logs/audio_dump) through the real BargeIn decision: for each stretch of caller speech (Silero VAD, the worker's settings)
that overlaps the agent being audible on the page (voice/heard.Playback fed the recorded output chunks), would it have interrupted, and why not."""
import asyncio
import csv
import sys
from pathlib import Path

import numpy as np
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.frames.frames import (
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)

from dealer_core.config import load_config
from dealer_core.voice.bargein import BargeIn
from dealer_core.voice.heard import Playback

DUMP = Path("/Users/fatima/dev/voice-agent/logs/audio_dump")
v = load_config(environ={}).voice


def rows(p):
    with open(p) as f:
        r = csv.reader(f)
        next(r)
        return [(float(a), int(b), int(c)) for a, b, c in r]


def events(p):
    out = []
    for line in open(p):
        t, kind, *rest = line.rstrip("\n").split("\t")
        out.append((float(t), kind, rest[0] if rest else ""))
    return out


async def one(stem_in, stem_out):
    ins, outs = rows(stem_in.with_suffix(".csv")), rows(stem_out.with_suffix(".csv"))
    pcm_in = np.fromfile(stem_in.with_suffix(".pcm"), dtype=np.int16)
    ev = events(stem_in.with_suffix(".txt")) if stem_in.with_suffix(".txt").exists() else []
    now = [0.0]
    pb = Playback(clock=lambda: now[0])
    pb.sentence("x")
    bi = BargeIn(pb, min_speech_secs=v.barge_in_min_speech_secs, level=v.barge_in_level)
    vad = SileroVADAnalyzer(sample_rate=16000, params=VADParams(confidence=v.vad_confidence, start_secs=v.vad_start_secs, stop_secs=0.35, min_volume=v.vad_min_volume))
    vad.set_sample_rate(16000)
    oi, pos, state, seg = 0, 0, VADState.QUIET, None
    segs = []
    for t, nbytes, rate in ins:
        while oi < len(outs) and outs[oi][0] <= t:
            ot, ob, orate = outs[oi]
            pb.audio(ob / 2 / orate, at=ot)
            oi += 1
        now[0] = t
        chunk = pcm_in[pos:pos + nbytes // 2]
        pos += nbytes // 2
        frame = InputAudioRawFrame(chunk.tobytes(), 16000, 1)
        new = await vad.analyze_audio(chunk.tobytes())
        if new == VADState.SPEAKING and state != VADState.SPEAKING:
            seg = {"t0": t, "audible_at_start": pb.playing(t), "audible_any": False, "voiced": 0.0, "voiced_audible": 0.0, "rms": [], "cut": False}
            bi.observe(VADUserStartedSpeakingFrame())
        if new == VADState.QUIET and state != VADState.QUIET and seg:
            bi.observe(VADUserStoppedSpeakingFrame())
            seg["t1"] = t
            segs.append(seg)
            seg = None
        state = new
        if seg is not None:
            x = chunk.astype(np.float32) / 32768
            r = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
            seg["rms"].append(r)
            a = pb.playing(t)
            seg["audible_any"] |= a
            if r >= v.barge_in_level:
                seg["voiced"] += len(x) / 16000
                if a:
                    seg["voiced_audible"] += len(x) / 16000
            if bi.observe(frame):
                seg["cut"] = True
    for s in segs:
        if not s["audible_any"]:
            continue
        words = " | ".join(w for (t, k, w) in ev if k == "stt" and s["t0"] - 0.2 <= t <= s.get("t1", s["t0"]) + 1.5)
        why = ("interrupted" if s["cut"] else
               "began before the agent was audible" if not s["audible_at_start"] else
               f"too little loud speech while audible ({s['voiced_audible']:.2f} s >= {v.barge_in_level})")
        print(f"  {s['t0'] % 1000:8.2f}  {s.get('t1', s['t0']) - s['t0']:4.2f} s  peak {max(s['rms']):.3f}  median {np.median(s['rms']):.3f}  {why:52s} {words[:60]}")
    return segs


async def main():
  allsegs = []
  for p in sorted(DUMP.glob("*_in.pcm")):
    stem_in = p.with_suffix("")
    stamp = stem_in.name[:15]
    outs = sorted(DUMP.glob(stamp[:13] + "*_out.pcm"))
    if not outs:
        continue
    print(stem_in.name)
    allsegs += [s for s in await one(stem_in, outs[0].with_suffix("")) if s["audible_any"]]
    sys.stdout.flush()
  n = len(allsegs); cut = sum(s["cut"] for s in allsegs); late = sum((not s["cut"]) and not s["audible_at_start"] for s in allsegs)
  print(f"TOTAL over the agent: {n}; interrupted {cut}; began before audible {late}; too little loud speech {n - cut - late}")

asyncio.run(main())
