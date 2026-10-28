"""A live status line for the console, so you can SEE the call working: it never prints what anyone said, only the state of the call."""
from __future__ import annotations

import time

from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, TranscriptionFrame, TTSSpeakFrame,
                                   VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TurnClock:
    """Times a turn from what the pipeline sees: when the caller stopped, when the words were understood, when the first reply sentence was ready,
    and when the first sound started. Reports once per turn through `on_timing(total_seconds, parts)`.

    `on_answer(seconds)` (optional) is told once per turn when the ANSWER started: the same clock, but a filler said first ("Let me check.",
    "Sure,": `fillers`) does not count, so a slow model is not hidden behind a quick filler. The worker's call cap follows this one."""

    def __init__(self, stop_secs, on_timing, clock=time.monotonic, on_answer=None, fillers: frozenset[str] = frozenset()) -> None:
        # `stop_secs` is the silence the voice detector waits before it reports the end of speech, or a function that says what it is right now: tap mode waits
        # 0.6 s (a longer 0.9 s after a phone-number question), not Smart Turn's 0.2 s. With the wrong number every time was 0.4 s too short (28 Oct review, B1).
        self.stop_secs, self.on_timing, self._now = stop_secs, on_timing, clock
        self._turn_stop = stop_secs if not callable(stop_secs) else 0.0           # the value that applied to the turn in progress
        self.on_answer, self.fillers = on_answer, fillers
        self.t_stop = self.t_stt = self.t_sentence = None
        self._turn: dict | None = None                                 # this turn's answer timing: base, filler first, answer ready, voice delay

    def see(self, frame: Frame, stage: str) -> None:
        now = self._now()
        if stage == "vad" and isinstance(frame, VADUserStoppedSpeakingFrame):
            self.t_stop, self.t_stt, self.t_sentence = now, None, None
            self._turn_stop = self.stop_secs() if callable(self.stop_secs) else self.stop_secs
            self._turn = {"base": now - self._turn_stop, "filler_first": None, "answer": None, "voice": None}
        elif stage == "stt" and isinstance(frame, TranscriptionFrame):
            self.t_stt = now
        elif stage == "brain" and isinstance(frame, TTSSpeakFrame):
            filler = frame.text.strip() in self.fillers
            if self.t_sentence is None:
                self.t_sentence = now
                if self._turn is not None:
                    self._turn["filler_first"] = filler
            if not filler and self._turn is not None and self._turn["answer"] is None:
                self._turn["answer"] = now
                self._report_answer()
        elif isinstance(frame, BotStartedSpeakingFrame) and self.t_stop is not None:
            parts = {"waiting for you to finish": self._turn_stop}
            if self.t_stt is not None:
                parts["speech recognition"] = self.t_stt - self.t_stop
                if self.t_sentence is not None:
                    parts["reply"] = self.t_sentence - self.t_stt
                    parts["voice"] = now - self.t_sentence
            total = now - (self.t_stop - self._turn_stop)               # from the real end of the caller's speech
            self.t_stop = None                                         # one report per turn
            if self._turn is not None and self.t_sentence is not None:
                self._turn["voice"] = now - self.t_sentence           # how long the voice took to start the first sentence
                if not self._turn["filler_first"]:
                    self._turn["answer"] = self._turn["answer"] or self.t_sentence
                self._report_answer()
            self.on_timing(total, parts)

    def _report_answer(self) -> None:
        """Once per turn, when both the answer's readiness and the voice's start-up time are known."""
        t = self._turn
        if t is None or t["answer"] is None or t["voice"] is None or self.on_answer is None:
            return
        self._turn = None
        self.on_answer(t["answer"] - t["base"] + t["voice"])


class StatusLine(FrameProcessor):
    def __init__(self, show: set[str], out=print, clock: TurnClock | None = None, stage: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._show, self._out, self._clock, self._stage = show, out, clock, stage

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._clock is not None:
            self._clock.see(frame, self._stage)
        if "vad" in self._show and isinstance(frame, VADUserStartedSpeakingFrame):
            self._out("  [hearing you]", flush=True)
        elif "vad" in self._show and isinstance(frame, VADUserStoppedSpeakingFrame):
            self._out("  [you stopped; working on it]", flush=True)
        elif "stt" in self._show and isinstance(frame, TranscriptionFrame):
            self._out(f"  [understood {len(frame.text.split())} words]", flush=True)           # the words themselves are never printed
        elif "bot" in self._show and isinstance(frame, BotStartedSpeakingFrame):
            self._out("  [agent speaking]", flush=True)
        elif "bot" in self._show and isinstance(frame, BotStoppedSpeakingFrame):
            self._out("  [your turn]", flush=True)
        await self.push_frame(frame, direction)
