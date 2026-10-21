"""A live status line for the console, so you can SEE the call working: it never prints what anyone said, only the state of the call."""
from __future__ import annotations

from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, TranscriptionFrame, VADUserStartedSpeakingFrame,
                                   VADUserStoppedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class StatusLine(FrameProcessor):
    def __init__(self, show: set[str], out=print, **kwargs) -> None:
        super().__init__(**kwargs)
        self._show, self._out = show, out

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
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
