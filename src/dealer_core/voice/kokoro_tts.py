"""Kokoro-82M (Apache-2.0) as Wazir's voice, run with OpenVINO on the laptop's CPU. A drop-in for PiperSherpa (same methods).

Measured on this laptop (bench_kokoro.py): ONNX Runtime ~0.5 s for a short phrase, OpenVINO CPU ~0.44 s (battery) and a real-time
factor of 0.16 (it makes speech ~6x faster than it plays). The iGPU (an unsupported resize op) and the NPU (variable output length) cannot
run it. So that the first sound does not wait on a whole sentence:

  * the first piece of every reply is short (a few words), the rest follows while it plays
  * phrases Wazir says all the time (greeting, "anything else?", booking questions) are rendered once and kept: they play at once

Phonemes come from kokoro-onnx (espeak-ng, the same GPL-3 data Piper uses: the same open licence question, docs/LICENSES.md).
"""
from __future__ import annotations

import queue
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .tts import pronounce

VOICES = {"michael": "am_michael", "adam": "am_adam", "puck": "am_puck", "eric": "am_eric", "liam": "am_liam", "onyx": "am_onyx",
          "george": "bm_george", "lewis": "bm_lewis", "fenrir": "am_fenrir",
          "heart": "af_heart", "bella": "af_bella", "sarah": "af_sarah", "emma": "bf_emma"}


class _OVSession:
    """What kokoro-onnx expects from an onnxruntime session, run on OpenVINO (one infer request per call: safe from many threads)."""

    def __init__(self, model_path: Path, cache_dir: Path) -> None:
        import onnxruntime as rt
        import openvino as ov

        core = ov.Core()
        self.compiled = core.compile_model(core.read_model(str(model_path)), "CPU",
                                           {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": str(cache_dir)})
        self._meta = rt.InferenceSession(str(model_path), providers=["CPUExecutionProvider"]).get_modelmeta()
        self._model_path = str(model_path)

    def run(self, _names, inputs):
        req = self.compiled.create_infer_request()
        req.infer({"tokens": inputs["tokens"], "style": inputs["style"], "speed": inputs["speed"]})
        return [req.get_output_tensor(0).data.copy()]

    def get_modelmeta(self):
        return self._meta

    def get_inputs(self):
        return [_Named("tokens", "tensor(int64)"), _Named("style", "tensor(float)"), _Named("speed", "tensor(float)")]

    def get_outputs(self):
        return [_Named("audio", "tensor(float)")]


class _Named:
    def __init__(self, name: str, kind: str) -> None:
        self.name, self.type = name, kind


def first_piece_split(text: str, first_words: int = 4, max_words: int = 12) -> list[str]:
    """'Sure, the RAV4 is a 2021 in white, with ...' -> ['Sure, the RAV4', 'is a 2021 in white,', 'with ...']: a short first piece so the
    first sound comes quickly; then clauses (at commas) of up to `max_words`."""
    words = text.split()
    if len(words) <= first_words + 2:
        return [text]
    cut = next((i + 1 for i, w in enumerate(words[:first_words + 3]) if w.endswith(",") and i >= 1), first_words)
    head, rest = " ".join(words[:cut]), words[cut:]
    out, cur = [head], []
    for w in rest:
        cur.append(w)
        if (w.endswith(",") and len(cur) >= 4) or len(cur) >= max_words:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


class KokoroOV:
    sample_rate = 24000

    def __init__(self, models: Path, voice: str = "sarah", speed: float = 1.0, cache_size: int = 300) -> None:
        from kokoro_onnx import Kokoro

        d = models / "kokoro-v1.0"
        self.kokoro = Kokoro.from_session(_OVSession(d / "kokoro-v1.0.onnx", models / "ov_cache"), str(d / "voices-v1.0.bin"))
        self.voice = VOICES.get(voice, voice)
        self.speed = speed
        self.name = f"kokoro:{self.voice}"
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def _say(self, piece: str) -> np.ndarray:
        key = piece.strip()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        audio, _ = self.kokoro.create(pronounce(key), voice=self.voice, speed=self.speed, lang="en-us")
        audio = np.asarray(audio, dtype=np.float32)
        if len(key.split()) <= 14:                                # keep short, reusable pieces (questions, booking prompts, greetings)
            with self._lock:
                self._cache[key] = audio
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return audio

    def preload(self, phrases: list[str]) -> float:
        """Render the phrases Wazir says all the time, before the first call (they then play with no wait)."""
        t0 = time.perf_counter()
        for p in phrases:
            for piece in first_piece_split(p):
                self._say(piece)
        return time.perf_counter() - t0

    def warm_up(self) -> None:
        self._say("Warm up.")

    def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        return np.concatenate([self._say(p) for p in first_piece_split(text)])

    def stream(self, text: str, speed: float = 1.0) -> Iterator[np.ndarray]:
        """Piece by piece: the short first piece is ready quickly; the next ones are made while it plays."""
        q: queue.Queue = queue.Queue()

        def run():
            try:
                for piece in first_piece_split(text):
                    q.put(self._say(piece))
            finally:
                q.put(None)

        threading.Thread(target=run, daemon=True).start()
        while (chunk := q.get()) is not None:
            yield chunk

    @staticmethod
    def to_pcm16(samples: np.ndarray) -> bytes:
        return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    def first_audio_ms(self, text: str) -> float:
        t0 = time.perf_counter()
        next(self.stream(text))
        return (time.perf_counter() - t0) * 1000
