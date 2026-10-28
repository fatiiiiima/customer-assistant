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

from .stt import SAMPLE_RATE, WhisperOpenVino, is_degenerate
from .tts import PiperSherpa


MIN_RMS = 0.005                      # loudest 0.3 s below this (about -46 dBFS) is room noise, not speech; Whisper invents words from it.
                                     # Measured on the real recordings: Whisper still hears them at 5-10% volume, so the gate must not be stricter.


def segment_rms(pcm16: bytes, window: int = 4800) -> float:
    """Level of the loudest 0.3 s: the segment also holds the 1 s of quiet before the speech, which would drag a whole-segment average down."""
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    if len(x) <= window:
        return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    return float(np.sqrt(np.convolve(x * x, np.ones(window) / window, "valid").max()))


SPEAK_UP = "Sorry, I can barely hear you. Could you speak up or move closer to the microphone?"

# What the recogniser writes for sound it cannot make out (Whisper learned from video captions): measured 28 Oct on quiet versions of the owner's own turns,
# "Please subscribe and hit the bell icon...", "Please feel free to ask in the comments below.", a lone "Okay." or "Thank you." for a whole sentence.
_INVENTED = re.compile(r"\b(?:subscribe|bell icon|comments? below|in the comments|thanks? for watching|thank you for (?:watching|your time|listening)|see you (?:next|in the next)|"
                       r"like and share|feel free to (?:ask|comment|post|call)|share this video|don't forget to)\b", re.I)
_LONE_FILLER = re.compile(r"^\s*(?:okay|ok|thank you|thanks|you|bye|so)\s*[.!?]*\s*$", re.I)


def too_quiet_to_trust(text: str, level: float, very_quiet: float, suspect_below: float) -> bool:
    """True when the turn should not be acted on, because the caller was too quiet for the words to be trusted: the level is under `very_quiet`, or it is under
    `suspect_below` and the words look invented. 0 turns a part off. Everything louder is left alone, whatever the words."""
    if not text or level is None:
        return False
    if very_quiet and level < very_quiet:
        return True
    return bool(suspect_below and level < suspect_below and (_INVENTED.search(text) or _LONE_FILLER.match(text) or is_degenerate(text, 0.0)))


UNCLEAR = "um"      # given to the agent when the recognizer produced a loop instead of words: the engine answers a lone filler with "Sorry, I didn't catch that."


def squeeze_punctuation(text: str) -> str:
    """'No!!!!!!!!' -> 'No!': a mark repeated 3+ times is Whisper looping, and as one 'word' it would pass every word-based guard."""
    return re.sub(r"([^\w\s])\1{2,}", r"\1", text)


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
    if text and is_degenerate(text, 0.0):
        return UNCLEAR
    text = collapse_repeats(squeeze_punctuation(text)) if text else ""
    return "" if text and is_phantom(text) else text


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower())]


def echoes_agent(heard: str, agent_line: str) -> bool:
    """True when `heard` is the agent's own question given back: it starts like one of the agent's sentences and is mostly made of its words (at least 5 words).
    Seen 28 Oct with quiet speech: the recognizer is given what the agent just said as a hint, and when the audio is weak it writes the hint down, so "I want to come and
    see it" became "Would you like to come and see it?"."""
    h = _words(heard)
    if len(h) < 5:
        return False
    for sentence in re.split(r"(?<=[.!?])\s+", agent_line):
        s = _words(sentence)
        if len(s) >= 3 and h[:3] == s[:3] and sum(w in s for w in h) / len(h) >= 0.7:
            return True
    return False


class EchoGuard:
    """Around a recogniser for one call. If the words it writes are the agent's own question, it listens again without the agent's sentence in the hint (the car names stay);
    if that is still the agent's question, nothing was understood and the caller is asked to repeat. Everything else is passed through unchanged."""

    def __init__(self, engine, agent_line, plain_prompt) -> None:
        self._engine, self._agent_line, self._plain_prompt = engine, agent_line, plain_prompt
        self.retries = 0

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def transcribe(self, pcm16: bytes, prompt: str | None = None) -> tuple[str, float]:
        text, ms = self._engine.transcribe(pcm16, prompt)
        line = self._agent_line() if self._agent_line else ""
        if not (text and line and echoes_agent(text, line)):
            return text, ms
        self.retries += 1
        again, ms2 = self._engine.transcribe(pcm16, self._plain_prompt())
        logger.debug("STT wrote the agent's own question: listened again without it in the hint")
        return (again if again and not echoes_agent(again, line) else UNCLEAR), ms + ms2


class OpenVinoWhisperSTTService(SegmentedSTTService):
    """Transcribes each VAD-segmented utterance with OpenVINO Whisper (CPU). Segments arrive as raw 16-bit 16 kHz mono PCM."""

    def __init__(self, *, engine: WhisperOpenVino, min_rms: float = MIN_RMS, **kwargs) -> None:
        self.min_rms = min_rms
        self.prompt_for = None                   # set by the pipeline: () -> the recognizer hint for what the agent just asked
        self.speculation = None                  # voice.speculate.Speculation: words already recognised during Smart Turn's hold
        self.on_text = None                      # diagnostics only (voice/audiotap.py): told the words of each segment
        super().__init__(sample_rate=SAMPLE_RATE, ttfs_p99_latency=0.3, settings=STTSettings(model=engine.name, language=Language.EN), **kwargs)
        self._engine = engine
        self.last_ms = 0.0                       # timing of the most recent transcription, read by the timing probe
        self.last_level: float | None = None     # loudest-0.3 s level of the segment just recognised (read by the brain: a quiet segment cannot hang up the call)

    @property
    def wants_wav_segments(self) -> bool:
        return False

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        rms = segment_rms(audio)
        self.last_level = rms
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
        if text and is_degenerate(text, len(audio) / 2 / SAMPLE_RATE):
            logger.debug(f"STT returned a loop ({len(text)} characters): the caller is asked to repeat")
            text = UNCLEAR
        if text:
            text = collapse_repeats(squeeze_punctuation(text))
        if text and is_phantom(text):
            logger.debug(f"STT dropped a repeated-word hallucination ({len(text.split())} words)")
            text = ""
        if text:
            # The transcript itself is never logged (privacy): only its length.
            logger.debug(f"STT {self.last_ms:.0f} ms, {len(text.split())} words, {len(audio) / 2 / SAMPLE_RATE:.1f} s of audio, level {rms:.3f}")
            if self.on_text:
                self.on_text(text)
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
