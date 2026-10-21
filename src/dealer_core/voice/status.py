"""A live status line for the console, so you can SEE the call working: it never prints what anyone said, only the state of the call."""
from __future__ import annotations

import time

from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, TranscriptionFrame, TTSSpeakFrame,
                                   VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TurnClock:
    """Times a turn from what the pipeline sees: when the caller stopped, when the words were understood, when the first reply sentence was ready,
    and when the first sound started. Reports once per turn through `on_timing(total_seconds, parts)`."""

    def __init__(self, stop_secs: float, on_timing, clock=time.monotonic) -> None:
        self.stop_secs, self.on_timing, self._now = stop_secs, on_timing, clock
        self.t_stop = self.t_stt = self.t_sentence = None

    def see(self, frame: Frame, stage: str) -> None:
        now = self._now()
        if stage == "vad" and isinstance(frame, VADUserStoppedSpeakingFrame):
            self.t_stop, self.t_stt, self.t_sentence = now, None, None
        elif stage == "stt" and isinstance(frame, TranscriptionFrame):
            self.t_stt = now
        elif stage == "brain" and isinstance(frame, TTSSpeakFrame) and self.t_sentence is None:
            self.t_sentence = now
        elif isinstance(frame, BotStartedSpeakingFrame) and self.t_stop is not None:
            parts = {"waiting for you to finish": self.stop_secs}
            if self.t_stt is not None:
                parts["speech recognition"] = self.t_stt - self.t_stop
                if self.t_sentence is not None:
                    parts["reply"] = self.t_sentence - self.t_stt
                    parts["voice"] = now - self.t_sentence
            total = now - (self.t_stop - self.stop_secs)               # from the real end of the caller's speech
            self.t_stop = None                                         # one report per turn
            self.on_timing(total, parts)


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
