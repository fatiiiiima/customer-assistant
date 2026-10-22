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


MIN_RMS = 0.012                      # below this (about -38 dBFS) a segment is room noise, not speech; Whisper invents words from it


def segment_rms(pcm16: bytes) -> float:
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def is_phantom(text: str) -> bool:
    """Whisper's typical inventions on noise: one word repeated ("Fiat, Fiat, Fiat") or a lone fragment made only of punctuation."""
    words = re.findall(r"[\w']+", text.lower())
    return not words or (len(words) >= 3 and len(set(words)) == 1)


class OpenVinoWhisperSTTService(SegmentedSTTService):
    """Transcribes each VAD-segmented utterance with OpenVINO Whisper (CPU). Segments arrive as raw 16-bit 16 kHz mono PCM."""

    def __init__(self, *, engine: WhisperOpenVino, min_rms: float = MIN_RMS, **kwargs) -> None:
        self.min_rms = min_rms
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
        text, self.last_ms = await asyncio.to_thread(self._engine.transcribe, audio)
        await self.stop_processing_metrics()
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
