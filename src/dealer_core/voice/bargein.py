"""Barge-in: the caller starts talking while the agent is still speaking, so the agent stops and listens.

Sits right after the voice-activity detector (and Smart Turn). When it sees "the caller started speaking" while the agent's audio is playing, it broadcasts an
InterruptionFrame through the pipeline: the voice stops synthesizing, queued audio is dropped (the website is told to stop playing), and the brain abandons the
reply it was still producing. The caller's words then go through speech recognition as a normal turn.

Needs echo control: with a headset, or in the browser (echo cancellation), the agent's own voice does not reach the microphone. On open laptop speakers the
agent would interrupt itself, so there it stays off (voice.barge_in / --no-barge-in).
"""
from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, VADUserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class BargeIn(FrameProcessor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._bot_speaking = False
        self.interruptions = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, VADUserStartedSpeakingFrame) and direction is FrameDirection.DOWNSTREAM and self._bot_speaking:
            self._bot_speaking = False
            self.interruptions += 1
            logger.debug("barge-in: the caller spoke over the agent")
            await self.broadcast_interruption()
        await self.push_frame(frame, direction)
