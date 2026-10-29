"""Barge-in: the caller starts talking while the agent is still speaking, so the agent stops and listens.

Sits right after the voice-activity detector (and Smart Turn). When the caller has really been speaking (`min_speech_secs` of audio above `level`, after the
detector's own start) while the agent is audible, it broadcasts an InterruptionFrame through the pipeline: the voice stops synthesizing, queued audio is dropped (the
website is told to stop playing), and the brain abandons the reply it was still producing and keeps only what the caller heard (voice/heard.py). The caller's words
then go through speech recognition as a normal turn.

29 Oct (owner: "the barge in is wrong"): it used to stop at the detector's first 0.1 s of sound, so a cough, an "mm-hmm" or the agent's own echo cut it off, and it
went by the server's "bot speaking", which ends half way through what the caller hears (the server sends audio at twice its speed); now `Playback.playing()`.

Needs echo control: with a headset, or in the browser (echo cancellation), the agent's own voice does not reach the microphone. On open laptop speakers the
agent would interrupt itself, so there it stays off (voice.barge_in / --no-barge-in).
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, InputAudioRawFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class BargeIn(FrameProcessor):
    def __init__(self, playback=None, min_speech_secs: float = 0.3, level: float = 0.01, **kwargs) -> None:
        super().__init__(**kwargs)
        self.playback, self.min_speech_secs, self.level = playback, min_speech_secs, level
        self._bot_speaking = False                     # the server's own view (used only without a Playback)
        self._candidate = False                        # the caller started speaking over the agent: is it real speech?
        self._voiced = 0.0
        self.interruptions = 0
        self.ignored = 0                               # starts over the agent that were too short to be speech (a cough, a click, a bit of echo)

    def agent_audible(self) -> bool:
        return self.playback.playing() if self.playback is not None else self._bot_speaking

    def observe(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> bool:
        """The decision, without the pipeline: True when this frame makes it an interruption."""
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif direction is not FrameDirection.DOWNSTREAM:
            return False
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._candidate, self._voiced = self.agent_audible(), 0.0
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._candidate:
                self.ignored += 1
            self._candidate = False
        elif isinstance(frame, InputAudioRawFrame) and self._candidate:
            if not self.agent_audible():
                self._candidate = False                # the agent finished by itself: nothing to interrupt
                return False
            x = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768
            if len(x) and float(np.sqrt(np.mean(x ** 2))) >= self.level:
                self._voiced += len(x) / max(1, frame.sample_rate * frame.num_channels)
            if self._voiced >= self.min_speech_secs:
                self._candidate, self._bot_speaking = False, False
                self.interruptions += 1
                return True
        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self.observe(frame, direction):
            logger.debug("barge-in: the caller spoke over the agent")
            await self.broadcast_interruption()
        await self.push_frame(frame, direction)
