"""Run audio FILES through the real Pipecat voice pipeline at real-time pace and time every stage.

No microphone or speaker needed: a file source plays the caller (20 ms frames, then silence), a recorder plays the speaker. Per turn it reports

  endpointing   end of the caller's speech -> VAD says "stopped"
  STT           VAD stopped -> transcript
  brain         transcript -> reply text (intent + tools + model-written reply)
  TTS first     reply text -> first audio sample
  TOTAL         end of the caller's speech -> first audio the caller would hear
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np
from pipecat.frames.frames import (EndFrame, Frame, InputAudioRawFrame, TranscriptionFrame, TTSAudioRawFrame, TTSSpeakFrame,
                                   VADUserStoppedSpeakingFrame)
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from .pipeline import STT_RATE, VoiceParts, build_pipeline

FRAME_MS = 20


class Probe(FrameProcessor):
    """Records when frames of interest pass (monotonic seconds). Frames are passed on unchanged."""

    def __init__(self, name: str, log: list[tuple[float, str, str]], kinds: tuple[type, ...], **kwargs) -> None:
        super().__init__(**kwargs)
        self._name, self._log, self._kinds = name, log, kinds

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, self._kinds):
            self._log.append((time.monotonic(), self._name, type(frame).__name__))
        await self.push_frame(frame, direction)


class FileSource(FrameProcessor):
    """Plays 16 kHz int16 mono clips as the caller would speak them: real-time frames, a pause for the reply, next clip."""

    def __init__(self, clips: list[np.ndarray], log: list, reply_wait: asyncio.Event, last_audio: list | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.clips, self._log, self._reply_wait = clips, log, reply_wait
        self._last_audio = last_audio if last_audio is not None else [0.0]      # monotonic time of the latest reply audio
        self.played = 0                                                          # clips fully sent so far

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)                      # StartFrame, EndFrame and the rest must travel on

    async def start_playing(self) -> None:
        silence = np.zeros(STT_RATE * FRAME_MS // 1000, dtype=np.int16)

        async def send(samples: np.ndarray) -> None:
            await self.push_frame(InputAudioRawFrame(audio=samples.tobytes(), sample_rate=STT_RATE, num_channels=1))
            await asyncio.sleep(FRAME_MS / 1000)

        await asyncio.sleep(0.5)
        for i, clip in enumerate(self.clips):
            self._reply_wait.clear()
            n = STT_RATE * FRAME_MS // 1000
            for k in range(0, len(clip), n):
                await send(clip[k:k + n] if len(clip[k:k + n]) == n else np.pad(clip[k:k + n], (0, n - len(clip[k:k + n]))))
            self.played = i + 1
            self._log.append((time.monotonic(), "speech", f"end:{i}"))
            for _ in range(60):                                        # keep the line open with silence until the reply starts (or 30 s)
                for _ in range(25):
                    await send(silence)
                if self._reply_wait.is_set():
                    break
            await asyncio.sleep(1.0)
            for _ in range(300):                                       # let the reply play out completely (1 s of quiet) before the next caller turn
                if time.monotonic() - self._last_audio[0] > 1.0:
                    break
                await asyncio.sleep(0.1)


@dataclass
class TurnTiming:
    index: int
    endpointing_ms: float = 0.0
    stt_ms: float = 0.0
    brain_ms: float = 0.0
    tts_first_ms: float = 0.0
    total_ms: float = 0.0
    transcript_words: int = 0
    note: str = ""


@dataclass
class OfflineResult:
    turns: list[TurnTiming] = field(default_factory=list)
    brain_log: list[dict] = field(default_factory=list)
    events: list[tuple[str, str]] = field(default_factory=list)       # ("YOU" | "AGENT", text) in order, as the live transcript would show
    transcripts: list[str] = field(default_factory=list)              # kept in memory for the caller of the harness only; never logged


async def run_offline(app, parts: VoiceParts, clips: list[np.ndarray], *, stop_secs: float = 0.35, speak_opening: bool = False,
                      timeout_s: float = 240.0, smart_turn: bool = False, smart_fallback_secs: float = 1.0, smart_short_answer_secs: float | None = None,
                      smart_peek: bool | None = None) -> OfflineResult:
    log: list[tuple[float, str, str]] = []
    reply_wait = asyncio.Event()
    last_audio = [0.0]
    transcripts: list[str] = []
    events: list[tuple[str, str]] = []

    class Recorder(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, TTSAudioRawFrame):
                last_audio[0] = time.monotonic()
                answered_this_clip = sum(1 for who, _ in events if who == "YOU") >= source.played          # audio before this clip's transcript is the previous reply's tail
                if answered_this_clip and not reply_wait.is_set():
                    log.append((time.monotonic(), "audio", "first"))
                    reply_wait.set()
            await self.push_frame(frame, direction)

    kinds = {"vad": (VADUserStoppedSpeakingFrame,), "stt": (TranscriptionFrame,), "brain": (TTSSpeakFrame,), "tts": (TTSAudioRawFrame,)}
    probes = {name: Probe(name, log, k) for name, k in kinds.items()}
    source = FileSource(clips, log, reply_wait, last_audio)
    pipeline, brain = build_pipeline(app, parts, source, Recorder(), stop_secs=stop_secs, speak_opening=speak_opening, probes=probes, smart_turn=smart_turn, smart_fallback_secs=smart_fallback_secs, smart_short_answer_secs=smart_short_answer_secs, smart_peek=smart_peek,
                                     on_turn=lambda r: transcripts.append(r.reply),
                                     on_heard=lambda t: events.append(("YOU", t)), on_say=lambda t: events.append(("AGENT", t)))
    worker = PipelineWorker(pipeline, params=PipelineParams(audio_in_sample_rate=STT_RATE, audio_out_sample_rate=parts.tts_engine.sample_rate),
                            idle_timeout_secs=None)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    async def play():
        await asyncio.sleep(1.0)
        await source.start_playing()
        await asyncio.sleep(3.0)
        await worker.queue_frame(EndFrame())

    player = asyncio.create_task(play())
    try:
        await asyncio.wait_for(runner.run(), timeout=timeout_s)
    finally:
        player.cancel()

    # reduce the event log into per-turn timings
    result = OfflineResult(brain_log=list(brain.turn_log), transcripts=transcripts, events=events)
    result.turns = reduce_log(log)
    return result


def reduce_log(log: list[tuple[float, str, str]]) -> list[TurnTiming]:
    """Per-turn stage timings from the event log. The stages of a turn are matched IN ORDER, each at or after the stage that feeds it
    (speech end -> VAD stop -> transcript -> first sentence -> first audio), so no stage can be attributed to a time before the one before it.
    (The old version took the first event of each kind inside a time window: a long reply still being spoken when the next caller clip began put
    the previous turn's sentences and audio in the new window, giving negative brain times and 0.7 s 'responses' that were really the old reply.)"""
    ends = sorted(t for t, n, k in log if n == "speech")
    ev = {name: sorted(t for t, n, k in log if n == name) for name in ("vad", "stt", "brain", "tts")}
    turns: list[TurnTiming] = []
    floor = {name: 0.0 for name in ev}
    for i, t_end in enumerate(ends):
        nxt = ends[i + 1] if i + 1 < len(ends) else float("inf")

        def after(name: str, t0: float, limit: float = float("inf")):
            return next((t for t in ev[name] if t >= max(t0, floor[name]) and t < limit), None)

        vad = after("vad", t_end, nxt)
        stt = after("stt", vad, nxt) if vad is not None else None
        brn = after("brain", stt, nxt) if stt is not None else None
        tts = after("tts", brn, nxt) if brn is not None else None
        for name, t in (("vad", vad), ("stt", stt), ("brain", brn), ("tts", tts)):
            if t is not None:
                floor[name] = t + 1e-6                                   # an event belongs to one turn only
        tt = TurnTiming(i)
        if vad is not None:
            tt.endpointing_ms = (vad - t_end) * 1000
        if stt is not None:
            tt.stt_ms = (stt - vad) * 1000
        if brn is not None:
            tt.brain_ms = (brn - stt) * 1000
        if tts is not None:
            tt.tts_first_ms = (tts - brn) * 1000
            tt.total_ms = (tts - t_end) * 1000
        else:
            tt.note = "no reply audio"
        turns.append(tt)
    return turns
