"""Text-to-speech: Piper (VITS) through sherpa-onnx, fp32, on the CPU (the Phase 0b winner; int8 and Kokoro are slower on this CPU).

Default voice: Piper `joe` (CC0). The model folder also holds espeak-ng phoneme data: espeak-ng is GPL-3, acceptable for an internal prototype
only (open licence decision, see docs/LICENSES.md).
"""
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

VOICES = {"joe": "vits-piper-en_US-joe-medium", "libritts_r": "vits-piper-en_US-libritts_r-medium", "ljspeech": "vits-piper-en_US-ljspeech-medium",
          "alba": "vits-piper-en_GB-alba-medium", "amy": "vits-piper-en_US-amy-low-int8"}


def voice_dir(models: Path, voice: str) -> Path:
    d = models / VOICES.get(voice, voice)
    nested = d / d.name                                  # the downloads unpack to <name>/<name>/
    return nested if nested.exists() else d


class PiperSherpa:
    def __init__(self, models: Path, voice: str = "joe", threads: int = 4) -> None:
        import sherpa_onnx

        d = voice_dir(models, voice)
        cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=str(next(d.glob("*.onnx"))), tokens=str(d / "tokens.txt"),
                                                       data_dir=str(d / "espeak-ng-data")), num_threads=threads, provider="cpu"))
        self.tts = sherpa_onnx.OfflineTts(cfg)
        self.sample_rate: int = self.tts.sample_rate
        self.name = f"piper:{voice}"

    def warm_up(self) -> None:
        self.tts.generate("Warm up.", sid=0, speed=1.0)

    def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        """The whole utterance as float32 samples."""
        return np.asarray(self.tts.generate(text, sid=0, speed=speed).samples, dtype=np.float32)

    def stream(self, text: str, speed: float = 1.0) -> Iterator[np.ndarray]:
        """Chunks of float32 samples as the model produces them (first chunk after about one sentence of work)."""
        q: queue.Queue = queue.Queue()

        def cb(samples, progress):                       # called by sherpa on its own thread, once per generated piece
            q.put(np.asarray(samples, dtype=np.float32))
            return 1                                     # 1 = keep going

        def run():
            try:
                self.tts.generate(text, sid=0, speed=speed, callback=cb)
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
