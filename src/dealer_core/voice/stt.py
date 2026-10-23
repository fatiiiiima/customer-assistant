"""Speech-to-text: OpenVINO GenAI Whisper on the CPU, biased with the car terms (the Phase 0b winner).

STT must NOT share the iGPU with the language model (measured: sharing slows the intent call from 0.9 s to 1.6 s), so the default device is CPU.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def hotword_terms(path: Path) -> list[str]:
    """The car makes/models and place names used as a Whisper initial prompt (one per line, # comments allowed)."""
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


class WhisperOpenVino:
    def __init__(self, model_dir: Path, cache_dir: Path, device: str = "CPU", hotwords: list[str] | None = None, language: str = "en") -> None:
        import openvino_genai as ov

        self.name = f"openvino-whisper:{model_dir.name}@{device}"
        self.pipe = ov.WhisperPipeline(str(model_dir), device, CACHE_DIR=str(cache_dir))
        self._lock = threading.Lock()
        self.cfg = self.pipe.get_generation_config()
        if not model_dir.name.split("-int")[0].endswith(".en"):        # English-only models refuse a language or task token
            self.cfg.language = f"<|{language}|>"
            self.cfg.task = "transcribe"
        if hotwords and hasattr(self.cfg, "initial_prompt"):
            self.cfg.initial_prompt = ", ".join(hotwords[:60])

    def warm_up(self) -> None:
        self.transcribe_float(np.zeros(SAMPLE_RATE, dtype=np.float32))              # compiles / loads the first time

    def transcribe_float(self, audio: np.ndarray, prompt: str | None = None) -> str:
        with self._lock:                                                          # one pipeline shared by every call: it is not re-entrant
            if prompt is not None and hasattr(self.cfg, "initial_prompt"):
                self.cfg.initial_prompt = prompt                                  # what to expect right now (see voice/prompts.py)
            res = self.pipe.generate(audio.astype(np.float32).tolist(), self.cfg)
        return str(res.texts[0] if hasattr(res, "texts") else res).strip()

    def transcribe(self, pcm16: bytes, prompt: str | None = None) -> tuple[str, float]:
        """16 kHz mono signed 16-bit PCM -> (text, milliseconds)."""
        t0 = time.perf_counter()
        text = self.transcribe_float(np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0, prompt)
        return text, (time.perf_counter() - t0) * 1000
