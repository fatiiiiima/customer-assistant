"""The Pipecat processor around the dialogue engine: a transcript in, the agent's spoken reply (text) out.

Everything between "what the caller said" and "what the agent says" is the DialogueEngine: intent (rules or model), tools, pricing engine,
state machine, model-written reply, safety checks. This processor only moves text in and out and records timings.
"""
from __future__ import annotations

import asyncio
import threading
import time

from loguru import logger
from pipecat.frames.frames import BotStartedSpeakingFrame, Frame, StartFrame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..dialogue.machine import DialogueEngine
from .services import SPEAK_UP, too_quiet_to_trust


class DealerBrain(FrameProcessor):
    def __init__(self, engine: DialogueEngine, *, speak_opening: bool = True, on_turn=None, on_heard=None, on_say=None, level_of=None, quiet_below: float = 0.0, speak_up: tuple[float, float] = (0.0, 0.0), **kwargs) -> None:
        super().__init__(**kwargs)
        self.engine, self.session = engine, engine.new_session()
        self._speak_opening, self._on_turn = speak_opening, on_turn
        self._on_heard, self._on_say = on_heard, on_say          # transcript hooks: what the caller said / what the agent says (off unless the caller of the pipeline asks)
        self._level_of, self._quiet_below = level_of, quiet_below   # () -> the level of the segment just recognised; below `quiet_below` the words may not end the call
        self._speak_up = speak_up                       # (very quiet, suspect) levels: see services.too_quiet_to_trust; (0, 0) = never
        self._asked_to_speak_up = 0                     # in a row: after two the words are used as they are, so a caller who cannot be louder is not stuck
        self.spoken_up = 0                              # turns answered with "speak up" (for tests and the log)
        self.turn_log: list[dict] = []                 # per turn: timings and flags only (no transcripts, no phone numbers)
        self._busy = asyncio.Lock()
        self.interrupted = 0                            # turns abandoned because the caller spoke over the agent
        self.speculation = None                         # voice.speculate.Speculation: a decision made during Smart Turn's hold
        # Tap mode keeps the microphone open until the reply is heard (voice/micgate.py ContinuationGuard). If the caller carries on speaking before then, the not-yet-heard
        # reply is dropped and what they said so far is joined to what comes next, so a sentence cut by a long pause is still one sentence.
        self._open_turn: str | None = None              # the words of the last turn whose reply the caller has not started to hear
        self._carry = ""                                 # words to put in front of the next turn
        self._defer_next = False                        # the next transcript is the first half of a sentence that is still being said: keep it, do not answer it
        self._abort = False                             # the turn being answered is to be dropped and carried (see take_back)
        self.continued = 0                              # turns joined to the next one (for tests and the log)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            if self._speak_opening:
                opening = self.engine.opening(self.session)
                if self._on_say:
                    self._on_say(opening)
                await self.push_frame(TTSSpeakFrame(opening, append_to_context=False))
            return
        if isinstance(frame, TranscriptionFrame):
            if self._defer_next:
                self._defer_next = False
                self._carry = f"{self._carry} {frame.text}".strip()
                self.continued += 1
                return                                              # the caller is still speaking: wait for the rest of the sentence
            await self._answer(frame.text)
            return
        if isinstance(frame, BotStartedSpeakingFrame):
            self._open_turn = None                                  # the reply is being heard now: it can no longer be taken back
        await self.push_frame(frame, direction)

    def turn_ended(self) -> None:
        """A turn has just ended: a first half that never produced a transcript must not swallow this one."""
        self._defer_next = False

    async def take_back(self) -> bool:
        """The caller started speaking again before the reply to their last turn was heard. Drops that reply and keeps the words to join to what comes next.
        Returns True when something may already be on its way to the voice, so the caller of this must interrupt the pipeline to throw it away."""
        text = self._open_turn
        if text is None:                                            # the transcript of the last turn has not arrived yet: keep it when it does
            self._defer_next = True
            return False
        self._carry = text                                          # (already includes anything carried into it)
        self._open_turn = None
        self.continued += 1
        if self._busy.locked():                                     # still being answered: the answer is cancelled and not remembered (see _answer)
            self._abort = True
            return True
        history = self.session.history                              # already answered, and the reply is queued for the voice: forget the exchange
        if len(history) >= 2 and history[-2] == ("customer", text):
            del history[-2:]
        return True

    async def _answer(self, text: str) -> None:
        """Runs the turn in a worker thread; every sentence goes to the TTS the moment the engine has verified it."""
        async with self._busy:                                      # one turn at a time per caller
            self._abort = False
            if self._carry:                                         # the first half of a sentence the caller carried on after
                text, self._carry = f"{self._carry} {text}".strip(), ""
            level = self._level_of() if self._level_of else None
            if self._asked_to_speak_up < 2 and too_quiet_to_trust(text, level, *self._speak_up):
                self._asked_to_speak_up += 1            # the words are not acted on and not remembered: they are probably not what was said
                self.spoken_up += 1
                if self._on_say:
                    self._on_say(SPEAK_UP)
                await self.push_frame(TTSSpeakFrame(SPEAK_UP, append_to_context=False))
                logger.debug(f"brain: too quiet to trust (level {level:.3f}), asked the caller to speak up")
                return
            self._asked_to_speak_up = 0
            self._open_turn = text
            heard = False
            t0 = time.perf_counter()
            loop = asyncio.get_running_loop()
            sentences: asyncio.Queue = asyncio.Queue()
            early = self.speculation.decision_for(text) if self.speculation is not None else None
            extra = {"predecided": early} if early is not None else {}
            if self._quiet_below and level is not None and level < self._quiet_below and getattr(self.engine, "guards_quiet_turns", False):
                extra["uncertain"] = True

            cancelled = threading.Event()

            def work() -> None:
                try:
                    for item in self.engine.handle_stream(self.session, text, **extra):
                        if cancelled.is_set():
                            break                                 # closing the generator also stops the model
                        loop.call_soon_threadsafe(sentences.put_nowait, item)
                finally:
                    loop.call_soon_threadsafe(sentences.put_nowait, None)

            worker = loop.run_in_executor(None, work)
            first_ms, result, said = 0.0, None, []
            try:
                while (item := await sentences.get()) is not None:
                    if isinstance(item, str):
                        if not heard:
                            heard = True
                            if self._on_heard:
                                self._on_heard(text)                # shown when the turn is real: a half sentence that gets taken back never appears
                        if not first_ms:
                            first_ms = (time.perf_counter() - t0) * 1000
                        said.append(item)
                        if self._on_say:
                            self._on_say(item)
                        await self.push_frame(TTSSpeakFrame(item, append_to_context=False))
                    else:
                        result = item
            except asyncio.CancelledError:                       # the caller spoke over us (barge-in): drop the rest of this reply
                cancelled.set()
                if self._abort:                                    # ...before hearing any of it: nothing is kept, the words are joined to the next turn
                    self._abort, self._carry = False, text
                    self._open_turn = None
                    raise
                self.session.remember("customer", text)           # the engine saves a turn only when it finishes: keep what really happened
                self.session.remember("agent", (" ".join(said) + " (interrupted)") if said else "(interrupted)")
                self.interrupted += 1
                raise
            await worker
            if not heard and self._on_heard:
                self._on_heard(text)                                # a turn with nothing to say still shows what the caller said
            if result is not None:
                d = result.debug
                self.turn_log.append({"first_sentence_ms": first_ms, "engine_ms": (time.perf_counter() - t0) * 1000, "intent": d.intent,
                                      "via": d.via.split(":")[0], "model_written": bool(d.draft), "risk": d.risk, "ended": result.ended, "fallback": "FALLBACK" in d.note,
                                      "note": d.note})            # the note holds reasons and flags only: no transcript, no phone number
                logger.debug(f"brain first sentence {first_ms:.0f} ms, intent {d.intent}, via {self.turn_log[-1]['via']}")
                if self._on_turn:
                    self._on_turn(result)
