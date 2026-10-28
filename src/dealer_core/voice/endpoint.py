"""Smart Turn end-of-turn detection (Pipecat's bundled smart-turn v3 model, BSD-2, runs locally on the CPU).

Plain voice-activity detection ends your turn after a fixed silence: short enough to feel quick and it cuts you off when you pause to think; long enough
to wait for you and every reply is slower. Smart Turn listens to HOW the sentence ended (falling tone, a finished clause) and decides "done" or
"still going". So the silence can be short (VAD stop ~0.2 s), and when the model says "still going" the turn stays open until you speak again or a
longer silence (`fallback_secs`) passes.

This processor sits right after the VAD. It lets the VAD's "stopped speaking" through only when Smart Turn says the turn is complete (or the fallback
silence has passed), and swallows the "started speaking" that follows a held stop, so the speech recognizer hears one continuous segment.

With a `speculation` (voice/speculate.py) a held stop also starts recognising the turn so far, and the model's decision, in the background:
used as they are if the fallback silence ends the turn, dropped if the caller speaks again.

EARLY START (`early_start_secs`): the same work also starts inside every turn, after that much real quiet (judged by loudness against the turn's own peak, not by the voice
detector, which waits for its full stop time: 0.6 s in tap mode). If the caller speaks again it is dropped; if the turn then ends, it is used as it is, and the answer is
that much sooner. The cost is the wasted work at every pause that was not the end.
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.frames.frames import Frame, InputAudioRawFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


QUIET_FLOOR = 0.01          # a 20 ms frame below this loudness (0-1) is quiet whatever the turn's peak was
QUIET_RATIO = 0.06          # ...and below this share of the turn's loudest frame (about 24 dB down: a soft last word is still louder)
MAX_EARLY_STARTS = 3        # per turn: each one is a recognition run that is wasted if the caller was only pausing


class SmartEndpointer(FrameProcessor):
    def __init__(self, sample_rate: int = 16000, fallback_secs: float = 0.6, analyzer: LocalSmartTurnAnalyzerV3 | None = None, short_answer_secs: float = 1.4, early_start_secs: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.early_start_secs = early_start_secs  # 0 = off: work starts only when Smart Turn holds a pause
        self._peak, self._quiet_n = 0.0, 0         # the loudest frame of this turn so far, and the length (in samples) of the quiet that has just gone by
        self._early = False                       # work was started in this quiet and the caller has not spoken since
        self._early_in_turn = 0
        self.early_starts = 0                     # how many times it was started (for tests and the log)
        self.short_answer_secs = short_answer_secs
        self.asked_question = None                # () -> bool: did the agent's last turn end on a question? (set by the pipeline; used by the length rule only)
        self.peek = None                          # async (pcm16 bytes) -> the words so far (the speech recognizer, on the audio of this turn)
        self.is_answer = None                     # (text) -> bool: a complete answer to what the agent just asked?
        self._turn_audio = bytearray()            # the audio of the turn in progress, for the peek and the speculation
        self._pre_roll = bytearray()              # the last 0.5 s before speech was detected (the detector fires ~0.1 s into the first word)
        self.speculation = None                   # voice.speculate.Speculation: work started during a hold
        self._turn_speech = 0.0                   # seconds of speech since the last finished turn
        self.short_answers = 0                    # turns finished early (length rule or words rule)
        self.peeked = 0                           # how many times the words were checked
        self._analyzer = analyzer or LocalSmartTurnAnalyzerV3(sample_rate=sample_rate, params=SmartTurnParams(stop_secs=fallback_secs))
        self._analyzer.set_sample_rate(sample_rate)
        self._speaking = False
        self._holding = False                     # a stop was held back: the turn is still open
        self.held = 0                             # how many stops Smart Turn held back (for the status line / tests)
        self.last_stop: VADUserStoppedSpeakingFrame | None = None

    def _reset_early(self) -> None:
        self._peak, self._quiet_n = 0.0, 0
        self._early, self._early_in_turn = False, 0

    def _watch_for_quiet(self, frame: InputAudioRawFrame) -> None:
        """One 20 ms frame of a turn in progress: start the work after `early_start_secs` of quiet, drop it when the sound comes back."""
        x = np.frombuffer(frame.audio, dtype="<i2")
        if not len(x):
            return
        loud = float(np.sqrt(np.mean((x.astype(np.float32) / 32768.0) ** 2)))
        self._peak = max(self._peak, loud)
        if loud >= max(QUIET_FLOOR, QUIET_RATIO * self._peak):
            self._quiet_n = 0
            if self._early:
                self._early = False
                self.speculation.resumed()                  # the caller was only pausing
            return
        self._quiet_n += len(x)                                             # counted in samples: ten 20 ms frames added up as seconds are 0.19999999999999998
        if not self._early and self._quiet_n >= self.early_start_secs * frame.sample_rate and len(self._turn_audio) > 3200 and self._early_in_turn < MAX_EARLY_STARTS:
            self._early = True
            self._early_in_turn += 1
            self.early_starts += 1
            self.speculation.start(bytes(self._turn_audio))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction is not FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._speaking = True
            if self._holding:                     # the caller carried on: the same turn continues
                self._holding = False
                if self.speculation is not None:
                    self.speculation.resumed()
                return
            self._reset_early()
            if not self._turn_audio:
                self._turn_audio += self._pre_roll
            if self.speculation is not None:
                self.speculation.clear()          # a new turn: nothing prepared for an earlier one applies
            await self.push_frame(frame, direction)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speaking = False
            self.last_stop = frame
            state, _ = await self._analyzer.analyze_end_of_turn()
            if state is not EndOfTurnState.COMPLETE and self.peek is not None and self.is_answer is not None and len(self._turn_audio) > 3200:
                self.peeked += 1
                try:
                    if self.is_answer(await self.peek(bytes(self._turn_audio))):
                        state = EndOfTurnState.COMPLETE        # the words are a complete answer to the question just asked: nothing more to wait for
                        self.short_answers += 1
                        clear = getattr(self._analyzer, "clear", None)
                        if clear:
                            clear()
                except Exception as e:
                    logger.debug(f"smart turn: peek failed ({type(e).__name__}); keeping the hold")
            if (state is not EndOfTurnState.COMPLETE and self.short_answer_secs > 0 and self.asked_question is not None and self._turn_speech <= self.short_answer_secs
                    and self.asked_question()):
                state = EndOfTurnState.COMPLETE        # a short answer right after a question (yes, "Toyota", a number, a time) is finished: do not wait for the fallback
                self.short_answers += 1
                clear = getattr(self._analyzer, "clear", None)
                if clear:
                    clear()
            if state is EndOfTurnState.COMPLETE:
                if self._early and self.speculation is not None:
                    self.speculation.confirm()              # the quiet it was started in went on to the end: what was prepared applies
                self._reset_early()
                self._turn_speech = 0.0
                self._turn_audio.clear()
                self._holding = False
                await self.push_frame(frame, direction)
            else:
                self._holding = True
                self._early = False                         # the hold below starts its own work
                self.held += 1
                if self.speculation is not None and len(self._turn_audio) > 3200:
                    self.speculation.start(bytes(self._turn_audio))       # most held turns were finished: have the answer's groundwork ready
                logger.debug("smart turn: still going, holding the end of turn")
        elif isinstance(frame, InputAudioRawFrame):
            if self._speaking or self._holding:
                self._turn_audio += frame.audio                   # the turn's audio, for the peek (capped at 12 s)
                del self._turn_audio[:-12 * 32000]
            else:
                self._pre_roll += frame.audio
                del self._pre_roll[:-16000]
            if self._speaking:
                self._turn_speech += len(frame.audio) / 2 / 16000
            if self.early_start_secs > 0 and self.speculation is not None and self._speaking and not self._holding:
                self._watch_for_quiet(frame)
            state = self._analyzer.append_audio(frame.audio, self._speaking)
            await self.push_frame(frame, direction)
            if self._holding and state is EndOfTurnState.COMPLETE:       # the fallback silence passed: end the turn now
                self._holding = False
                if self.speculation is not None:
                    self.speculation.confirm()                            # no new speech since the hold: what was started applies
                self._turn_speech = 0.0
                self._turn_audio.clear()
                stop = self.last_stop
                await self.push_frame(VADUserStoppedSpeakingFrame(stop_secs=stop.stop_secs if stop else 0.0), direction)
        else:
            await self.push_frame(frame, direction)
