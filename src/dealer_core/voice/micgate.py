"""Tap-to-talk: the microphone is open only while the visitor is speaking their turn.

Flow (the website's default): the visitor taps the mic -> speaks -> the server decides the turn is over (VAD + Smart Turn) and CLOSES the gate -> the agent answers with the
microphone closed -> the visitor taps again to reply, or taps during the answer to interrupt it (barge-in by tap, no guessing from sound).

Why: with an always-open microphone the agent hears room noise (phantom words), its own voice on laptop speakers, and cuts people off mid-sentence; a closed microphone while the
agent talks removes all three, and an explicit tap is a reliable interruption.

Messages (browser -> server): {"type": "mic", "on": true | false}. Server -> browser: {"type": "mic", "on": false, "reason": "turn"} when the turn is over.
"""
from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, InputAudioRawFrame, InputTransportMessageFrame,
                                   VADUserStoppedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

SILENCE_FRAMES = 60                       # 1.2 s of silence fed in when the visitor closes the mic by tapping: lets the detector end the turn


class MicGate(FrameProcessor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.open = True
        self._bot_speaking = False
        self.taps_while_speaking = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, InputTransportMessageFrame):
            msg = frame.message
            if isinstance(msg, dict) and msg.get("type") == "mic":
                await self.set_open(bool(msg.get("on")))
                return
        elif isinstance(frame, InputAudioRawFrame) and direction is FrameDirection.DOWNSTREAM and not self.open:
            return                                                    # the microphone is closed: nothing gets through
        await self.push_frame(frame, direction)

    async def set_open(self, on: bool) -> None:
        if on and not self.open:
            self.open = True
            if self._bot_speaking:                                    # a tap while the agent speaks: stop it at once
                self.taps_while_speaking += 1
                logger.debug("tap-to-talk: the visitor tapped over the agent")
                await self.broadcast_interruption()
        elif not on and self.open:
            self.open = False
            for _ in range(SILENCE_FRAMES):                           # the visitor tapped to finish: give the detector the silence it needs to end the turn
                await self.push_frame(InputAudioRawFrame(audio=bytes(640), sample_rate=16000, num_channels=1), FrameDirection.DOWNSTREAM)

    def close_after_turn(self) -> None:
        self.open = False                                             # the server already decided the turn is over


class StopNotifier(FrameProcessor):
    """Calls `callback()` when the turn is over (the caller's 'stopped speaking' got through Smart Turn)."""

    def __init__(self, callback, **kwargs) -> None:
        super().__init__(**kwargs)
        self._callback = callback

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStoppedSpeakingFrame) and direction is FrameDirection.DOWNSTREAM:
            await self._callback()
        await self.push_frame(frame, direction)
