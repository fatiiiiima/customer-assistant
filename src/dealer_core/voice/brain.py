"""The Pipecat processor around the dialogue engine: a transcript in, the agent's spoken reply (text) out.

Everything between "what the caller said" and "what the agent says" is the DialogueEngine: intent (rules or model), tools, pricing engine,
state machine, model-written reply, safety checks. This processor only moves text in and out and records timings.
"""
from __future__ import annotations

import asyncio
import time

from loguru import logger
from pipecat.frames.frames import Frame, StartFrame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..dialogue.machine import DialogueEngine


class DealerBrain(FrameProcessor):
    def __init__(self, engine: DialogueEngine, *, speak_opening: bool = True, on_turn=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.engine, self.session = engine, engine.new_session()
        self._speak_opening, self._on_turn = speak_opening, on_turn
        self.turn_log: list[dict] = []                 # per turn: timings and flags only (no transcripts, no phone numbers)
        self._busy = asyncio.Lock()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            if self._speak_opening:
                await self.push_frame(TTSSpeakFrame(self.engine.opening(self.session), append_to_context=False))
            return
        if isinstance(frame, TranscriptionFrame):
            await self._answer(frame.text)
            return
        await self.push_frame(frame, direction)

    async def _answer(self, text: str) -> None:
        """Runs the turn in a worker thread; every sentence goes to the TTS the moment the engine has verified it."""
        async with self._busy:                                      # one turn at a time per caller
            t0 = time.perf_counter()
            loop = asyncio.get_running_loop()
            sentences: asyncio.Queue = asyncio.Queue()

            def work() -> None:
                try:
                    for item in self.engine.handle_stream(self.session, text):
                        loop.call_soon_threadsafe(sentences.put_nowait, item)
                finally:
                    loop.call_soon_threadsafe(sentences.put_nowait, None)

            worker = loop.run_in_executor(None, work)
            first_ms, result = 0.0, None
            while (item := await sentences.get()) is not None:
                if isinstance(item, str):
                    if not first_ms:
                        first_ms = (time.perf_counter() - t0) * 1000
                    await self.push_frame(TTSSpeakFrame(item, append_to_context=False))
                else:
                    result = item
            await worker
            if result is not None:
                d = result.debug
                self.turn_log.append({"first_sentence_ms": first_ms, "engine_ms": (time.perf_counter() - t0) * 1000, "intent": d.intent,
                                      "via": d.via.split(":")[0], "model_written": bool(d.draft), "risk": d.risk, "ended": result.ended,
                                      "note": d.note})            # the note holds reasons and flags only: no transcript, no phone number
                logger.debug(f"brain first sentence {first_ms:.0f} ms, intent {d.intent}, via {self.turn_log[-1]['via']}")
                if self._on_turn:
                    self._on_turn(result)
