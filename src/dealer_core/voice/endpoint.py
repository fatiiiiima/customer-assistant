"""Smart Turn end-of-turn detection (Pipecat's bundled smart-turn v3 model, BSD-2, runs locally on the CPU).

Plain voice-activity detection ends your turn after a fixed silence: short enough to feel quick and it cuts you off when you pause to think; long enough
to wait for you and every reply is slower. Smart Turn listens to HOW the sentence ended (falling tone, a finished clause) and decides "done" or
"still going". So the silence can be short (VAD stop ~0.2 s), and when the model says "still going" the turn stays open until you speak again or a
longer silence (`fallback_secs`) passes.

This processor sits right after the VAD. It lets the VAD's "stopped speaking" through only when Smart Turn says the turn is complete (or the fallback
silence has passed), and swallows the "started speaking" that follows a held stop, so the speech recognizer hears one continuous segment.
"""
from __future__ import annotations

from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.frames.frames import Frame, InputAudioRawFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class SmartEndpointer(FrameProcessor):
    def __init__(self, sample_rate: int = 16000, fallback_secs: float = 0.6, analyzer: LocalSmartTurnAnalyzerV3 | None = None, short_answer_secs: float = 1.4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.short_answer_secs = short_answer_secs
        self.asked_question = None                # () -> bool: did the agent's last turn end on a question? (set by the pipeline)
        self._turn_speech = 0.0                   # seconds of speech since the last finished turn
        self.short_answers = 0                    # turns finished early because the answer was short and followed a question
        self._analyzer = analyzer or LocalSmartTurnAnalyzerV3(sample_rate=sample_rate, params=SmartTurnParams(stop_secs=fallback_secs))
        self._analyzer.set_sample_rate(sample_rate)
        self._speaking = False
        self._holding = False                     # a stop was held back: the turn is still open
        self.held = 0                             # how many stops Smart Turn held back (for the status line / tests)
        self.last_stop: VADUserStoppedSpeakingFrame | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction is not FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._speaking = True
            if self._holding:                     # the caller carried on: the same turn continues
                self._holding = False
                return
            await self.push_frame(frame, direction)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speaking = False
            self.last_stop = frame
            state, _ = await self._analyzer.analyze_end_of_turn()
            if (state is not EndOfTurnState.COMPLETE and self.asked_question is not None and self._turn_speech <= self.short_answer_secs
                    and self.asked_question()):
                state = EndOfTurnState.COMPLETE        # a short answer right after a question (yes, "Toyota", a number, a time) is finished: do not wait for the fallback
                self.short_answers += 1
                clear = getattr(self._analyzer, "clear", None)
                if clear:
                    clear()
            if state is EndOfTurnState.COMPLETE:
                self._turn_speech = 0.0
                self._holding = False
                await self.push_frame(frame, direction)
            else:
                self._holding = True
                self.held += 1
                logger.debug("smart turn: still going, holding the end of turn")
        elif isinstance(frame, InputAudioRawFrame):
            if self._speaking:
                self._turn_speech += len(frame.audio) / 2 / 16000
            state = self._analyzer.append_audio(frame.audio, self._speaking)
            await self.push_frame(frame, direction)
            if self._holding and state is EndOfTurnState.COMPLETE:       # the fallback silence passed: end the turn now
                self._holding = False
                self._turn_speech = 0.0
                stop = self.last_stop
                await self.push_frame(VADUserStoppedSpeakingFrame(stop_secs=stop.stop_secs if stop else 0.0), direction)
        else:
            await self.push_frame(frame, direction)
