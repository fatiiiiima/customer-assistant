"""Work started during Smart Turn's hold, so it is already done when the turn ends.

When Smart Turn hears a pause but judges the sentence unfinished ("still going"), the turn stays open until the caller speaks again or the
fallback silence (`voice.smart_fallback_secs`, 0.6 s) passes. Most of the time the caller had in fact finished, and the agent then waited
0.6 s before even starting to recognise the words and ask the model what to do.

Now, the moment a stop is held, the turn's audio so far is transcribed and (agent engine) the model's tool decision is made, in the
background. Then:
  * the fallback silence passes with no new speech: the turn ends on exactly that audio, so the transcript is used as it is (no second
    recognition) and so is the decision, if the call has not moved on meanwhile;
  * the caller speaks again: both results are dropped; the next hold starts over on the whole turn.
The cost is the wasted work when the caller carries on; the decision is skipped when the governor says the model has no room.

One Speculation per call. `start` / `resumed` / `confirm` are called by the endpointer (voice/endpoint.py) on the event loop;
`transcript` by the recognizer service; `decision_for` by the brain.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Callable


class Speculation:
    def __init__(self, transcribe: Callable[[bytes], str], decide: Callable[[str], dict | None] | None = None,
                 version: Callable[[], int] | None = None) -> None:
        self.transcribe = transcribe              # (pcm16) -> text, blocking (runs in a thread)
        self.decide = decide                      # (text) -> the decision, or None when there is nothing to prepare; blocking
        self.version = version or (lambda: 0)     # changes whenever the call moves on (the session's history length)
        self._text: asyncio.Future | None = None
        self._decision: concurrent.futures.Future | None = None
        self._at_version = 0
        self._gen = 0
        self.valid = False                        # the held turn ended with no new speech: the results describe exactly this turn
        self.started = self.used = self.dropped = 0

    def start(self, pcm16: bytes) -> None:
        """A stop was held: start on the turn so far (replaces any earlier start for this turn)."""
        self._gen += 1
        gen, self.valid, self.started = self._gen, False, self.started + 1
        self._at_version = self.version()
        self._decision = concurrent.futures.Future() if self.decide else None
        decision = self._decision
        words: concurrent.futures.Future = concurrent.futures.Future()
        loop = asyncio.get_running_loop()

        def work() -> None:
            text = ""
            try:
                text = self.transcribe(pcm16)
                words.set_result(text)                        # the recognizer can have the words now, while the decision is still being made
            except Exception as e:
                words.set_exception(e)
            if decision is not None:                          # the decision follows the words, in the same thread
                try:
                    decision.set_result(self.decide(text) if text and gen == self._gen else None)
                except Exception:
                    decision.set_result(None)

        self._text = asyncio.wrap_future(words, loop=loop)
        loop.run_in_executor(None, work)

    def resumed(self) -> None:
        """The caller spoke again: what was started describes a turn that is not over."""
        if self._text is not None:
            self.dropped += 1
        self._gen += 1
        self._text, self._decision, self.valid = None, None, False

    def clear(self) -> None:
        """A new turn begins: nothing from an earlier one may be used (it may have ended without its words being asked for)."""
        self._gen += 1
        self._text, self._decision, self.valid = None, None, False

    def confirm(self) -> None:
        """The fallback silence passed with no new speech: the turn ends on the audio the work started from."""
        self.valid = self._text is not None

    async def transcript(self) -> str | None:
        """The words of the turn that just ended, if they were already being recognised; None means recognise them now."""
        if not self.valid or self._text is None:
            return None
        self.valid = False                                    # handed out once: a later turn never gets these words
        try:
            text = await self._text
        except Exception:
            return None
        self.used += 1
        return text

    def decision_for(self, text: str) -> concurrent.futures.Future | None:
        """The decision prepared for these words (wait on it with .result()), if it still applies: same words, the call has not moved on."""
        d, self._decision = self._decision, None
        if d is None or self._at_version != self.version() or self._text is None or not self._text.done():
            return None
        try:
            if self._text.result() != text:
                return None
        except Exception:
            return None
        return d
