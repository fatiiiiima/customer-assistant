"""A voice made by a streaming server (Chatterbox Turbo on the Mac's GPU: scripts/chatterbox_server.py), with Kokoro as the fallback (`tts.backend: chatterbox`).

Owner and their boss, 29 Oct: Chatterbox sounds better than Kokoro (section 38; samples in logs/voice_samples/for_boss). Each sentence is sent whole and its audio
passed on as it arrives (`POST {url}/audio/speech`, "stream": true, 16-bit mono 24 kHz in HTTP chunks), so there are no joins inside a sentence. If the server is
not there or sends nothing within `first_timeout_s`, the sentence is said by Kokoro (trimmed joins, same 24 kHz), so a call never goes silent; after a failure
the server is tried again on the next sentence. Sentences of up to 14 words are kept once made (questions and booking lines come back every call), and the
phrases said all the time are made in the background at start-up, so they play at once without slowing the start.

Pocket-TTS (`tts.backend: pocket`, the default since 30 Oct, section 42): the same class twice. Eve from scripts/pocket_server.py, whose fallback is alba from the
speech toolkit's server (a separate program, so one crashing leaves the other), whose fallback is Kokoro: a call goes silent only if all three fail.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Iterator

import numpy as np
from loguru import logger

from .tts import pronounce


class StreamingVoice:
    def __init__(self, url: str, fallback, model: str = "chatterbox-turbo", voice: str | None = None, first_timeout_s: float = 2.0, cache_size: int = 300,
                 client=None) -> None:
        import httpx

        self.base, self.model, self.voice, self.fallback = url.rstrip("/"), model, voice, fallback
        self.sample_rate = fallback.sample_rate                       # 24 kHz, as the server's
        self.first_timeout_s = first_timeout_s
        self.name = f"stream:{model}{'/' + voice if voice else ''}@{self.base} (fallback {fallback.name})"
        self.client = client or httpx.Client(timeout=httpx.Timeout(10.0, read=first_timeout_s))
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self._last_live = 0.0                                         # when a live sentence was last asked for (the background filling waits for quiet)
        self.fallbacks = 0                                            # sentences said by the fallback because the server failed (for the log and tests)
        self.served = 0

    def pieces(self, text: str) -> list[str]:
        return [text]                                                 # a sentence is made whole: no joins inside it

    def to_pcm16(self, samples: np.ndarray) -> bytes:
        return self.fallback.to_pcm16(samples)

    def warm_up(self) -> None:
        self.fallback.warm_up()
        try:
            for _ in self._remote("Hello."):
                pass
        except Exception as e:                                        # the worker still starts: Kokoro says everything until the server answers
            logger.warning(f"voice server {self.base} not reachable ({type(e).__name__}): {self.fallback.name} will speak until it is")

    def preload(self, phrases: list[str], first_only: list[str] = ()) -> float:
        """The server's versions of the phrases said all the time, made in the background and kept (Kokoro, the fallback, is made ready by load_engines)."""
        def fill():                                                   # in the background, giving way to live sentences (the server stops a "low" one for them)
            for p in phrases:
                while time.monotonic() - self._last_live < 3.0:        # a call is being spoken to: wait for a quiet moment
                    time.sleep(0.5)
                for _ in range(5):
                    try:
                        self._keep(p, np.concatenate(list(self._remote(p, background=True))))
                        break
                    except Exception:
                        time.sleep(5.0)                               # a live sentence came first: try again a little later
            logger.debug(f"voice: {len(self._cache)} phrases ready from the voice server")
        threading.Thread(target=fill, daemon=True, name="voice-preload").start()
        return 0.0

    def _keep(self, text: str, audio: np.ndarray) -> None:
        if len(text.split()) <= 14 and len(audio):
            with self._lock:
                self._cache[text] = audio
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)

    def _remote(self, text: str, background: bool = False) -> Iterator[np.ndarray]:
        """The server's audio for one sentence, chunk by chunk (raises if it fails before or while sending, or, made in the background, if it gave way to a live one)."""
        body = {"model": self.model, "input": pronounce(text), "stream": True, "response_format": "pcm", **({"voice": self.voice} if self.voice else {}),
                **({"priority": "low"} if background else {})}
        with self.client.stream("POST", f"{self.base}/audio/speech", json=body) as r:
            r.raise_for_status()
            rest = b""
            for data in r.iter_bytes():
                data, rest = rest + data, b""
                if len(data) % 2:
                    data, rest = data[:-1], data[-1:]
                if data:
                    yield np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    def stream(self, text: str, speed: float = 1.0) -> Iterator[np.ndarray]:
        key = text.strip()
        self._last_live = time.monotonic()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
        if hit is not None:
            yield hit
            return
        got: list[np.ndarray] = []
        try:
            for chunk in self._remote(key):
                got.append(chunk)
                yield chunk
        except Exception as e:
            if got:                                                   # cut in the middle of the sentence: what was said stands
                logger.warning(f"voice server stopped mid-sentence ({type(e).__name__})")
                return
            self.fallbacks += 1
            logger.warning(f"voice server {self.base} failed ({type(e).__name__}): {self.fallback.name} says this sentence")
            yield from self.fallback.stream(text)
            return
        self.served += 1
        if got:
            self._keep(key, np.concatenate(got))

    def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        return np.concatenate(list(self.stream(text)))

    def first_audio_ms(self, text: str) -> float:
        t0 = time.perf_counter()
        next(self.stream(text))
        return (time.perf_counter() - t0) * 1000
