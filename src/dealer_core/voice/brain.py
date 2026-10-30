"""The Pipecat processor around the dialogue engine: a transcript in, the agent's spoken reply (text) out.

Everything between "what the caller said" and "what the agent says" is the DialogueEngine: intent (rules or model), tools, pricing engine,
state machine, model-written reply, safety checks. This processor only moves text in and out and records timings.
"""
from __future__ import annotations

import asyncio
import re
import threading
import time

from loguru import logger
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, StartFrame, TranscriptionFrame, TTSSpeakFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_output import BOT_VAD_STOP_FALLBACK_SECS

from ..dialogue.machine import DialogueEngine
from .services import _INVENTED, SPEAK_UP, echoes_agent, too_quiet_to_trust


# A turn that is only "Actually." / "Wait," / "Hold on": the start of what the caller is about to say, often followed by a pause (29 Oct, a barge-in: "Actually," was
# answered on its own and "do you have any BMWs?" came as a turn of its own). Held up to MARKER_WAIT_S for the rest; "yes", "no" and "okay" are answers and are not held.
# Said over the agent, these do not stop it (voice/bargein.py LateBargeIn): a listener's "yes", "okay", "mm-hmm", a thank-you, or the recogniser's own fillers. Whole turns only.
_GOES_ALONG = re.compile(r"(?:(?:yes|yeah|yep|yup|ok|okay|sure|right|alright|all right|mm+|hmm+|mhm|mm-?hmm|uh-?huh|um+|uh+|ah+|oh|i see|got it|cool|nice|great|"
                         r"perfect|good|thanks|thank you|you|bye|so)[\s,.!?-]*)+", re.I)
_MARKER_ONLY = re.compile(r"(?:(?:actually|wait|hold on|hang on|um+|uh+|er+|erm|well|so|oh|hmm+|sorry|excuse me|one moment)[\s,.!?-]*)+", re.IGNORECASE)
MARKER_WAIT_S = 1.2

# Silence on the line (owner, 30 Oct: "user should start speaking"; "if the model doesn't hear anything, it should ask an are you still there question"; section 49)
STILL_THERE_ASK = "Hello? Are you still there?"
SILENT_GOODBYE = "I'll end the call here. Tap the microphone any time to carry on where we left off."
WELCOME_BACK = "Welcome back. Where were we?"


def _louder(a: float | None, b: float | None) -> float | None:
    """The higher of two levels, either of which may be unknown (None)."""
    return b if a is None else a if b is None else max(a, b)


class DealerBrain(FrameProcessor):
    def __init__(self, engine: DialogueEngine, *, speak_opening: bool = True, on_turn=None, on_heard=None, on_say=None, level_of=None, quiet_below: float = 0.0, speak_up: tuple[float, float] = (0.0, 0.0),
                 background_below: float = 0.0, first_turn_below: float = 0.0, greet_after: float = 0.0, still_there_after: float = 0.0, hang_up_after: float = 0.0,
                 welcome_back: bool = False, on_before_greeting=None, on_silent_end=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.engine, self.session = engine, engine.new_session()
        self._speak_opening, self._on_turn = speak_opening, on_turn
        self._on_heard, self._on_say = on_heard, on_say          # transcript hooks: what the caller said / what the agent says (off unless the caller of the pipeline asks)
        self._level_of, self._quiet_below = level_of, quiet_below   # () -> the level of the segment just recognised; below `quiet_below` the words may not end the call
        self._speak_up = speak_up                       # (very quiet, suspect) levels: see services.too_quiet_to_trust; (0, 0) = never
        self._asked_to_speak_up = 0                     # in a row: after two the words are used as they are, so a caller who cannot be louder is not stuck
        self.spoken_up = 0                              # turns answered with "speak up" (for tests and the log)
        self._background_below = background_below      # hands-free: a turn quieter than this share of the caller's usual level is someone else in the room (0 = off)
        self._caller_levels: list[float] = []           # levels of the last turns acted on (the caller's own voice is the loudest of them)
        self._first_turn_below = first_turn_below      # hands-free: before any turn is acted on, a turn quieter than this is the room (0 = off)
        self._room_in_a_row = 0                         # turns ignored by that bar in a row: the third gets a "speak up", so a caller who is really that quiet is not ignored for ever
        self.background = 0                             # turns ignored as background voices (for tests and the log)
        self.turn_log: list[dict] = []                 # per turn: timings and flags only (no transcripts, no phone numbers)
        self._busy = asyncio.Lock()
        self.interrupted = 0                            # turns abandoned because the caller spoke over the agent
        self.speculation = None                         # voice.speculate.Speculation: a decision made during Smart Turn's hold
        # Tap mode keeps the microphone open until the reply is heard (voice/micgate.py ContinuationGuard). If the caller carries on speaking before then, the not-yet-heard
        # reply is dropped and what they said so far is joined to what comes next, so a sentence cut by a long pause is still one sentence.
        self._open_turn: str | None = None              # the words of the last turn whose reply the caller has not started to hear
        self._carry = ""                                 # words to put in front of the next turn
        self._carry_level: float | None = None          # the loudest level among those words: a good first half must not be thrown away because of a quiet blip after it
        self._open_level: float | None = None           # the level of the turn whose reply has not been heard yet
        self._defer_next = False                        # the next transcript is the first half of a sentence that is still being said: keep it, do not answer it
        self._abort = False                             # the turn being answered is to be dropped and carried (see take_back)
        self.continued = 0                              # turns joined to the next one (for tests and the log)
        self._marker_timer = None                       # a turn that was only "Actually," is answered on its own if nothing follows within MARKER_WAIT_S
        self.marker_wait_s = MARKER_WAIT_S
        self._marker_waiting = False                    # the caller is speaking after a held "Actually,": wait for that transcript
        self._caller_speaking = False                   # between the detector's "started" and "stopped" speaking
        self.playback = None                            # voice.heard.Playback: told when a reply starts, so what the caller heard of it can be worked out
        self._cut_heard: str | None = None              # the caller interrupted the reply being written: what they had heard of it (see reply_cut)
        # Silence (section 49). The caller speaks first: the greeting (or, for a caller who came back, WELCOME_BACK) is said only after `greet_after` s of quiet. After the
        # agent has finished speaking, `still_there_after` s of quiet gets STILL_THERE_ASK; `hang_up_after` s more gets SILENT_GOODBYE, and once that has been heard the
        # call ends (`on_silent_end`). The clock stops while either side talks or a reply is being made, and starts again from the current stage after; a turn acted on
        # (or a "speak up") puts it back to the start. 0 = off (greet_after 0: greet at once, as before).
        self._greet_after, self._still_there_after, self._hang_up_after = greet_after, still_there_after, hang_up_after
        self._welcome_back = welcome_back
        self._on_before_greeting, self._on_silent_end = on_before_greeting, on_silent_end
        self._quiet_stage = "off"                       # greet | first | second | ending | off
        self._quiet_timer = None
        self._agent_speaking = False                    # between the output's "bot started" and "bot stopped speaking": the clock does not run
        # Our voices do not mark the end of a sentence, so the output reports "bot stopped speaking" when its queue has been empty for BOT_VAD_STOP_FALLBACK_SECS (3 s):
        # MEASURED on a test call, 30 Oct, "are you still there?" came 11.0 s after the agent's last word instead of 8. Taken off the waits that start then.
        self.stop_report_lag_s = float(BOT_VAD_STOP_FALLBACK_SECS)
        self.quiet_prompts: list[str] = []              # what the silence clock said (for tests and the log)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self.begin()
            return
        if isinstance(frame, TranscriptionFrame):
            if self._defer_next:
                self._defer_next = False
                self._carry = f"{self._carry} {frame.text}".strip()
                self._carry_level = _louder(self._carry_level, self._level_of() if self._level_of else None)
                self.continued += 1
                return                                              # the caller is still speaking: wait for the rest of the sentence
            if self._marker_timer is not None or self._marker_waiting:  # what follows a held "Actually,": the two are one turn (see _answer's carry)
                if self._marker_timer is not None:
                    self._marker_timer.cancel()
                self._marker_timer, self._marker_waiting = None, False
            elif _MARKER_ONLY.fullmatch(frame.text.strip()) and not self._carry:
                self._carry = frame.text.strip()
                self._carry_level = self._level_of() if self._level_of else None
                if self._caller_speaking:                           # already saying the rest (its start reached us before these words did): wait for it
                    self._marker_waiting = True
                else:
                    self._arm_marker(self.marker_wait_s)
                return
            await self._answer(frame.text)
            return
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._caller_speaking = True
            self._cancel_quiet()
            if self._marker_timer is not None:
                self._marker_timer.cancel()                         # the rest of the sentence after "Actually," is being said: its transcript will come
                self._marker_timer = None
                self._marker_waiting = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._caller_speaking = False
            self._arm_quiet()                                       # a reply cancels it when it starts; a turn that gets none (the room) leaves the clock running
            if self._marker_waiting:
                self._marker_waiting = False                        # ...and if it gives no transcript after all, "Actually," is answered alone, a little later
                self._arm_marker(self.marker_wait_s + 1.0)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._open_turn = None                                  # the reply is being heard now: it can no longer be taken back
            self._agent_speaking = True
            self._cancel_quiet()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._agent_speaking = False
            if self._quiet_stage == "ending":
                await self._end_for_silence()
            else:
                self._arm_quiet(already=self.stop_report_lag_s)
        await self.push_frame(frame, direction)

    # ------------------------------------------------------------------------------------------------ silence
    async def begin(self) -> None:
        """The call has started."""
        if (self._speak_opening or self._welcome_back) and self._greet_after:
            self._quiet_stage = "greet"                     # the caller first: greet only if they say nothing
            self._arm_quiet()
        elif self._speak_opening:
            await self._greet()
            self._quiet_stage = "first"
        else:
            self._quiet_stage = "first"                     # a call carried on after a drop: listen
            self._arm_quiet()

    def _quiet_wait(self) -> float:
        return {"greet": self._greet_after, "first": self._still_there_after, "second": self._hang_up_after, "ending": 12.0}.get(self._quiet_stage, 0.0)

    def _arm_quiet(self, already: float = 0.0) -> None:
        """`already`: seconds of the quiet that have passed before this was called."""
        self._cancel_quiet()
        wait = self._quiet_wait()
        if wait > 0 and (not self._agent_speaking or self._quiet_stage == "ending"):   # while the agent is audible, its "stopped speaking" arms it
            loop = asyncio.get_running_loop()
            self._quiet_timer = loop.call_later(max(0.0, wait - already), lambda: loop.create_task(self._quiet_reached()))

    def _cancel_quiet(self) -> None:
        if self._quiet_timer is not None:
            self._quiet_timer.cancel()
            self._quiet_timer = None

    def _heard_someone(self) -> None:
        """A turn was acted on, or the caller was asked to speak up: someone is there, so the clock starts again from the beginning."""
        if self._quiet_stage not in ("off", "ending"):
            self._quiet_stage = "first"
            self._arm_quiet()                                       # from the full wait (the reply, when it starts, stops the clock until it has been heard)

    async def _quiet_reached(self) -> None:
        self._quiet_timer = None
        if self._busy.locked() or self._caller_speaking or (self._agent_speaking and self._quiet_stage != "ending"):   # not silence
            self._arm_quiet()
            return
        stage = self._quiet_stage
        if stage == "greet":
            self._quiet_stage = "first"
            if self._on_before_greeting:
                await self._on_before_greeting()
            await self._greet()
        elif stage == "first":
            self._quiet_stage = "second"
            await self._say_quiet(STILL_THERE_ASK)
        elif stage == "second":
            self._quiet_stage = "ending"
            await self._say_quiet(SILENT_GOODBYE)
            self._arm_quiet()                                       # if the end of the goodbye is never reported, end anyway
        elif stage == "ending":
            await self._end_for_silence()

    async def _greet(self) -> None:
        if self.playback is not None:
            self.playback.new_reply()
        if self._welcome_back and not self._speak_opening:
            line = WELCOME_BACK
            self.session.remember("agent", line)
        else:
            line = self.engine.opening(self.session)
        self.quiet_prompts.append(line)
        if self._on_say:
            self._on_say(line)
        await self.push_frame(TTSSpeakFrame(line, append_to_context=False))

    async def _say_quiet(self, line: str) -> None:
        """Not remembered: the conversation goes on from where it was."""
        if self.playback is not None:
            self.playback.new_reply()
        self.quiet_prompts.append(line)
        if self._on_say:
            self._on_say(line)
        await self.push_frame(TTSSpeakFrame(line, append_to_context=False))

    async def _end_for_silence(self) -> None:
        self._cancel_quiet()
        self._quiet_stage = "off"
        logger.debug("brain: no answer to 'are you still there?': the call is ended (it can be carried on)")
        if self._on_silent_end:
            await self._on_silent_end()

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
        self._carry, self._carry_level = text, _louder(self._carry_level, self._open_level)       # (already includes anything carried into it)
        self._open_turn = None
        self.continued += 1
        if self._busy.locked():                                     # still being answered: the answer is cancelled and not remembered (see _answer)
            self._abort = True
            return True
        history = self.session.history                              # already answered, and the reply is queued for the voice: forget the exchange
        if len(history) >= 2 and history[-2] == ("customer", text):
            del history[-2:]
        return True

    def _arm_marker(self, wait_s: float) -> None:
        loop = asyncio.get_running_loop()
        self._marker_timer = loop.call_later(wait_s, lambda: loop.create_task(self._answer_held()))

    async def _answer_held(self) -> None:
        """Nothing followed a held "Actually,": answer it as it is."""
        self._marker_timer, self._marker_waiting = None, False
        if self._carry:
            await self._answer("")

    def reply_cut(self, heard: str) -> None:
        """The caller spoke over the agent: of its reply they heard `heard` (voice.heard.Playback). The call history keeps only that, so the next answer neither takes
        for granted what was not heard nor asks a question that never got said; of the cars just listed only those heard are kept for "the second one".
        29 Oct: the whole reply stayed in the history (it is saved when written, about 1 s in, and spoken for 5 to 10 s more)."""
        if not heard:
            return
        if self._busy.locked():                                     # still being written: the turn is saved when it is cancelled (see _answer)
            self._cut_heard = heard
            return
        history = self.session.history
        if history and history[-1][0] == "agent" and history[-1][1].startswith(heard.removesuffix(" ...")):    # (a "speak up" is not in the history: then nothing to correct)
            logger.debug(f"barge-in: the caller heard {len(heard.split())} of {len(history[-1][1].split())} words of the reply")    # counts only: no transcript in the log
            history[-1] = ("agent", f"{heard} (interrupted)")
            self._keep_heard_cars(heard)
            self.interrupted += 1

    def _is_background(self, level: float | None) -> bool:
        """A turn under `background_below` of the loudest of the last 8 turns acted on is someone else talking near the microphone. The loudest, not the
        median: in the owner's 30 Oct call the room was heard before the caller, and those turns must not set the bar. The owner's own turns in recorded
        calls: 0.10 to 0.27, the quietest 0.37 of the loudest (section 43). Before any turn is acted on there is nothing to compare with, so a turn under
        `first_turn_below` is the room (the owner's office call, 30 Oct: noise turns 0.024-0.040 before the caller spoke at 0.46; section 46)."""
        if level is None:
            return False
        if not self._caller_levels:
            return bool(self._first_turn_below and level < self._first_turn_below)
        return bool(self._background_below and level < self._background_below * max(self._caller_levels))

    def talks_over(self, text: str) -> bool:
        """A turn recognised while the agent is still audible, which BargeIn did not stop it for: is it the caller talking over the agent? Not when it is the room
        (`_is_background`), too quiet to trust, words the recogniser invents, the agent's own reply coming back through the microphone, or only "yes" / "okay" / "thank you"."""
        if not text.strip() or _GOES_ALONG.fullmatch(text.strip()) or _INVENTED.search(text):
            return False
        level = self._level_of() if self._level_of else None
        if self._is_background(level) or too_quiet_to_trust(text, level, *self._speak_up):
            return False
        playing = self.playback.text() if self.playback is not None else ""
        return not (playing and echoes_agent(text, playing))

    def _keep_heard_cars(self, heard: str) -> None:
        shown = self.session.shown
        named = [c for c in shown if c.model.lower() in heard.lower()]
        if named and len(named) < len(shown):
            self.session.shown = named

    async def _answer(self, text: str) -> None:
        """Runs the turn in a worker thread; every sentence goes to the TTS the moment the engine has verified it."""
        async with self._busy:                                      # one turn at a time per caller
            self._abort, self._cut_heard = False, None
            if self.playback is not None:
                self.playback.new_reply()
            level = self._level_of() if self._level_of else None
            if self._carry:                                         # the first half of a sentence the caller carried on after
                carried, carried_level = self._carry, self._carry_level
                self._carry, self._carry_level = "", None
                if too_quiet_to_trust(text, level, *self._speak_up) and not too_quiet_to_trust(carried, carried_level, *self._speak_up):
                    text, level = carried, carried_level            # 28 Oct, Test B: "I want to hear more about the Macan." was recognised, then a quiet blip (level 0.018) came while the microphone
                    logger.debug("brain: a too-quiet piece after a good first half was dropped")      # was held open, and the pair was refused as one: the first half is answered alone
                else:
                    text, level = f"{carried} {text}".strip(), _louder(level, carried_level)
            if self._is_background(level):              # 30 Oct, the owner's hands-free call: voices across the room were answered, or asked to speak up
                self.background += 1                    # not answered, not remembered
                if self._caller_levels:
                    logger.debug(f"brain: a turn at level {level:.3f}, under {self._background_below:.0%} of the caller's usual level, ignored as background")
                    return
                self._room_in_a_row += 1
                logger.debug(f"brain: a turn at level {level:.3f} before the caller has been heard, under {self._first_turn_below}, ignored as the room")
                if self._room_in_a_row < 3:
                    return
                self._room_in_a_row = 0                 # three in a row: perhaps a caller far from the microphone, so ask once (the words are still not used)
                self.spoken_up += 1
                self._heard_someone()
                if self._on_say:
                    self._on_say(SPEAK_UP)
                await self.push_frame(TTSSpeakFrame(SPEAK_UP, append_to_context=False))
                return
            self._room_in_a_row = 0
            if self._asked_to_speak_up < 2 and too_quiet_to_trust(text, level, *self._speak_up):
                self._asked_to_speak_up += 1            # the words are not acted on and not remembered: they are probably not what was said
                self.spoken_up += 1
                self._heard_someone()
                if self._on_say:
                    self._on_say(SPEAK_UP)
                await self.push_frame(TTSSpeakFrame(SPEAK_UP, append_to_context=False))
                logger.debug(f"brain: too quiet to trust (level {level:.3f}), asked the caller to speak up")
                return
            self._asked_to_speak_up = 0
            self._heard_someone()
            if level is not None:
                self._caller_levels = (self._caller_levels + [level])[-8:]
            self._open_turn, self._open_level = text, level
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
                    self._abort, self._carry, self._carry_level = False, text, level
                    self._open_turn = None
                    raise
                self.session.remember("customer", text)           # the engine saves a turn only when it finishes: keep what really happened
                heard_part = self._cut_heard if self._cut_heard is not None else " ".join(said)      # what was heard of it, when the playback is known
                self.session.remember("agent", f"{heard_part} (interrupted)" if heard_part else "(interrupted)")
                if self._cut_heard:
                    self._keep_heard_cars(self._cut_heard)
                self._cut_heard = None
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
                if result.ended:                                    # the caller said goodbye: no "are you still there?" after it
                    self._cancel_quiet()
                    self._quiet_stage = "off"
                if self._on_turn:
                    self._on_turn(result)
