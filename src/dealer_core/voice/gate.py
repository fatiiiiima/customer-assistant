"""Half-duplex gate: while the agent is speaking (and a moment after), the microphone is ignored.

Without this, a laptop microphone hears the laptop's own speaker, the VAD fires on the agent's voice and the agent answers itself.
Cost: the caller cannot interrupt (no barge-in yet). A headset removes the echo and allows `half_duplex=False`.
"""
from __future__ import annotations

import time

from loguru import logger

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class HalfDuplexGate(FrameProcessor):
    def __init__(self, tail_secs: float = 0.5, clock=time.monotonic, **kwargs) -> None:
        super().__init__(**kwargs)
        self._speaking, self._until, self._tail, self._clock = False, 0.0, tail_secs, clock

    def drops(self, frame: Frame) -> bool:
        """Update from the frame and say whether it must be dropped (microphone audio while the agent speaks or its echo fades)."""
        if isinstance(frame, BotStartedSpeakingFrame):
            self._speaking = True
            logger.debug('gate: agent started speaking, microphone muted')
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.debug('gate: agent stopped speaking')
            self._speaking, self._until = False, self._clock() + self._tail          # the room is still ringing for a moment
        elif isinstance(frame, InputAudioRawFrame):
            return self._speaking or self._clock() < self._until
        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not self.drops(frame):
            await self.push_frame(frame, direction)
