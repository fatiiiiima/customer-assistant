"""Tap-to-talk: the microphone is open only while the visitor is speaking their turn.

Flow (the website's default): the visitor taps the mic -> speaks -> the server decides the turn is over (VAD + Smart Turn) and CLOSES the gate -> the agent answers with the
microphone closed -> the visitor taps again to reply, or taps during the answer to interrupt it (barge-in by tap, no guessing from sound).

Why: with an always-open microphone the agent hears room noise (phantom words), its own voice on laptop speakers, and cuts people off mid-sentence; a closed microphone while the
agent talks removes all three, and an explicit tap is a reliable interruption.

Messages (browser -> server): {"type": "mic", "on": true | false}. Server -> browser: {"type": "mic", "on": false, "reason": "turn"} when the turn is over.
"""
from __future__ import annotations

import asyncio

from loguru import logger
from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, InputAudioRawFrame, InputTransportMessageFrame,
                                   VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

SILENCE_FRAMES = 60                       # 1.2 s of silence fed in when the visitor closes the mic by tapping: lets the detector end the turn


class MicGate(FrameProcessor):
    def __init__(self, start_open: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.open = start_open
        self._bot_speaking = False
        self.taps_while_speaking = 0
        self._pending = False                 # the turn is over but the microphone is kept open until the reply is heard (see hold_open_until_reply)
        self._timer = None
        self._on_close = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._close_now()                                         # the reply has begun: now the microphone closes
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
            self.cancel_pending()                                         # the visitor tapped to finish: no waiting
            self.open = False
            for _ in range(SILENCE_FRAMES):                           # the visitor tapped to finish: give the detector the silence it needs to end the turn
                await self.push_frame(InputAudioRawFrame(audio=bytes(640), sample_rate=16000, num_channels=1), FrameDirection.DOWNSTREAM)

    def close_after_turn(self) -> None:
        self.open = False                                             # the server already decided the turn is over

    @property
    def pending(self) -> bool:
        return self._pending

    def hold_open_until_reply(self, grace_s: float, on_close) -> None:
        """The turn is over, but keep listening until the reply starts to play (or `grace_s` passes): a caller who was only pausing can carry on and the two
        halves become one sentence (27 Oct: the page closed the microphone the moment the server decided, and the rest of the sentence was never heard).
        `on_close()` (a coroutine function) runs when the microphone does close, to tell the page."""
        self._cancel_timer()
        self._pending, self._on_close = True, on_close
        self._timer = asyncio.get_running_loop().call_later(grace_s, self._close_now)

    def cancel_pending(self) -> None:
        """The caller is speaking again: stay open, the closing is off."""
        self._pending, self._on_close = False, None
        self._cancel_timer()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _close_now(self) -> None:
        if not self._pending:
            return
        callback = self._on_close
        self.cancel_pending()
        self.open = False
        if callback is not None:
            asyncio.get_running_loop().create_task(callback())

    async def cleanup(self):
        await super().cleanup()
        self._cancel_timer()


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


class ContinuationGuard(FrameProcessor):
    """Tap mode: the turn was ended but the microphone is still open until the reply is heard. If the caller starts speaking in that window they were only pausing:
    the reply that was not yet heard is thrown away and what they said so far is kept to be joined to the rest (DealerBrain.take_back). Sits after Smart Turn, where
    the voice detector's "started" and "stopped" frames arrive."""

    def __init__(self, gate: MicGate, brain, **kwargs) -> None:
        super().__init__(**kwargs)
        self._gate, self._brain = gate, brain
        self.continued = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, VADUserStartedSpeakingFrame) and self._gate.pending:
                self._gate.cancel_pending()
                self.continued += 1
                logger.debug("tap-to-talk: the caller carried on before the reply was heard; the two halves become one turn")
                if await self._brain.take_back():
                    await self.broadcast_interruption()                   # drops whatever was already on its way to the voice
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                self._brain.turn_ended()
        await self.push_frame(frame, direction)
