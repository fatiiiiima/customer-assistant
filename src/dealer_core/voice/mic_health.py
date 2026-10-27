"""Tells the log when the microphone audio of a turn arrived damaged, so an intermittent "it did not hear me" can be caught as it happens.

Seen in the 27 Oct recordings (docs/ARCH_REVIEW.md section 7): in some turns about half of the real time never reached the server, as pieces of ~0.3 s
with ~0.4 s missing between them, and a turn could start with 0.7-1.0 s of exact digital silence. Clean turns arrive in a steady 20 ms cadence.
This sits after the voice detector and writes ONE line per turn, only when something is wrong. It reads no speech, keeps no audio.

  dropout  = the next 20 ms chunk came 0.1-1.0 s after the previous one (longer gaps are the microphone being closed between turns)
  silence  = chunks that are exactly zero (the browser delivered nothing real)
"""
from __future__ import annotations

import time

from loguru import logger
from pipecat.frames.frames import Frame, InputAudioRawFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

DROPOUT_MIN_S, DROPOUT_MAX_S = 0.10, 1.0
SILENCE_WARN_S = 0.30


class MicHealth(FrameProcessor):
    def __init__(self, clock=time.monotonic, **kwargs) -> None:
        super().__init__(**kwargs)
        self._clock, self._last = clock, None
        self.reset()

    def reset(self) -> None:
        self.chunks, self.dropouts, self.missing_s, self.silence_s = 0, 0, 0.0, 0.0

    def feed(self, audio: bytes, sample_rate: int = 16000) -> None:
        now, dur = self._clock(), len(audio) / 2 / sample_rate
        if self._last is not None and DROPOUT_MIN_S <= now - self._last <= DROPOUT_MAX_S:
            self.dropouts += 1
            self.missing_s += (now - self._last) - dur
        self._last = now
        self.chunks += 1
        if not any(audio):
            self.silence_s += dur

    def damaged(self) -> bool:
        return self.dropouts > 0 or self.silence_s >= SILENCE_WARN_S

    def finish_turn(self) -> str | None:
        """The line to log for the turn that just ended (None when it was healthy); starts the next turn's count."""
        line = (f"mic audio damaged in this turn: {self.dropouts} dropout(s), about {self.missing_s:.2f} s of it missing, "
                f"{self.silence_s:.2f} s of digital silence, {self.chunks} chunks") if self.damaged() else None
        self.reset()
        return line

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, InputAudioRawFrame):
                self.feed(frame.audio, frame.sample_rate)
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                line = self.finish_turn()
                if line:
                    logger.warning(line)
        await self.push_frame(frame, direction)
