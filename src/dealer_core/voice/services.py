"""Pipecat services around the local STT and TTS classes. No network, no API keys."""
from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator, AsyncIterator

import numpy as np
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from .stt import SAMPLE_RATE, WhisperOpenVino
from .tts import PiperSherpa


MIN_RMS = 0.005                      # loudest 0.3 s below this (about -46 dBFS) is room noise, not speech; Whisper invents words from it.
                                     # Measured on the real recordings: Whisper still hears them at 5-10% volume, so the gate must not be stricter.


def segment_rms(pcm16: bytes, window: int = 4800) -> float:
    """Level of the loudest 0.3 s: the segment also holds the 1 s of quiet before the speech, which would drag a whole-segment average down."""
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    if len(x) <= window:
        return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    return float(np.sqrt(np.convolve(x * x, np.ones(window) / window, "valid").max()))


def is_phantom(text: str) -> bool:
    """Whisper's typical inventions on noise: one word repeated ("Fiat, Fiat, Fiat") or a lone fragment made only of punctuation."""
    words = re.findall(r"[\w']+", text.lower())
    return not words or (len(words) >= 3 and len(set(words)) == 1)


def collapse_repeats(text: str) -> str:
    """Whisper sometimes loops on a phrase ("my name is, my name is, my name is, ..."): keep one copy of a phrase repeated 3+ times in a row."""
    tokens = text.split()
    key = [re.sub(r"[^\w']", "", t.lower()) for t in tokens]
    i, out = 0, []
    while i < len(tokens):
        for n in range(2, 5):                                     # phrases only: "five five five" in a phone number is real
            reps = 1
            while key[i:i + n] == key[i + reps * n:i + (reps + 1) * n] and len(key[i:i + n]) == n:
                reps += 1
            if reps >= 3:
                out += tokens[i:i + n]
                i += reps * n
                break
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out).strip(" ,")


def clean_transcript(text: str) -> str:
    """The recognizer's words as the agent receives them: loops collapsed, a repeated-word invention dropped (empty)."""
    text = collapse_repeats(text) if text else ""
    return "" if text and is_phantom(text) else text


class OpenVinoWhisperSTTService(SegmentedSTTService):
    """Transcribes each VAD-segmented utterance with OpenVINO Whisper (CPU). Segments arrive as raw 16-bit 16 kHz mono PCM."""

    def __init__(self, *, engine: WhisperOpenVino, min_rms: float = MIN_RMS, **kwargs) -> None:
        self.min_rms = min_rms
        self.prompt_for = None                   # set by the pipeline: () -> the recognizer hint for what the agent just asked
        self.speculation = None                  # voice.speculate.Speculation: words already recognised during Smart Turn's hold
        super().__init__(sample_rate=SAMPLE_RATE, ttfs_p99_latency=0.3, settings=STTSettings(model=engine.name, language=Language.EN), **kwargs)
        self._engine = engine
        self.last_ms = 0.0                       # timing of the most recent transcription, read by the timing probe

    @property
    def wants_wav_segments(self) -> bool:
        return False

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        rms = segment_rms(audio)
        if rms < self.min_rms:
            logger.debug(f"STT skipped: segment too quiet to be speech (level {rms:.4f})")
            return
        await self.start_processing_metrics()
        early = await self.speculation.transcript() if self.speculation is not None else None
        if early is not None:                    # recognised while Smart Turn waited, from this same audio: nothing left to do
            text, self.last_ms = early, 0.0
        else:
            text, self.last_ms = await asyncio.to_thread(self._engine.transcribe, audio, self.prompt_for() if self.prompt_for else None)
        await self.stop_processing_metrics()
        if text:
            text = collapse_repeats(text)
        if text and is_phantom(text):
            logger.debug(f"STT dropped a repeated-word hallucination ({len(text.split())} words)")
            text = ""
        if text:
            # The transcript itself is never logged (privacy): only its length.
            logger.debug(f"STT {self.last_ms:.0f} ms, {len(text.split())} words, {len(audio) / 2 / SAMPLE_RATE:.1f} s of audio, level {rms:.3f}")
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language.EN)


class SherpaPiperTTSService(TTSService):
    """Streams Piper speech as 16-bit PCM; the first chunk is ready after about one sentence of synthesis work."""

    def __init__(self, *, engine: PiperSherpa, **kwargs) -> None:
        super().__init__(sample_rate=engine.sample_rate, settings=TTSSettings(model=None, voice=engine.name, language=Language.EN), **kwargs)
        self._engine = engine

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        loop = asyncio.get_running_loop()
        it = self._engine.stream(text)

        async def chunks() -> AsyncIterator[bytes]:
            while True:
                chunk = await loop.run_in_executor(None, next, it, None)
                if chunk is None:
                    return
                yield self._engine.to_pcm16(chunk)

        try:
            await self.start_tts_usage_metrics(text)
            async for frame in self._stream_audio_frames_from_iterator(chunks(), in_sample_rate=self._engine.sample_rate, context_id=context_id):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as e:
            logger.error(f"{self} exception: {type(e).__name__}")
            yield ErrorFrame(error=f"TTS error: {type(e).__name__}")
        finally:
            await self.stop_ttfb_metrics()
