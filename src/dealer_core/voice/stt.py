"""Speech-to-text: OpenVINO GenAI Whisper on the CPU, biased with the car terms (the Phase 0b winner).

STT must NOT share the iGPU with the language model (measured: sharing slows the intent call from 0.9 s to 1.6 s), so the default device is CPU.
"""
from __future__ import annotations

import io
import queue
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def hotword_terms(path: Path) -> list[str]:
    """The car makes/models and place names used as a Whisper initial prompt (one per line, # comments allowed)."""
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


class WhisperOpenVino:
    """`copies` > 1 loads that many pipelines: the calls of one worker transcribe in parallel instead of queueing behind one lock
    (Whisper base is ~150 MB per copy; each copy uses its own CPU threads)."""

    def __init__(self, model_dir: Path, cache_dir: Path, device: str = "CPU", hotwords: list[str] | None = None, language: str = "en",
                 copies: int = 1) -> None:
        import openvino_genai as ov

        self.name = f"openvino-whisper:{model_dir.name}@{device}" + (f"x{copies}" if copies > 1 else "")
        self._free: queue.Queue = queue.Queue()
        for _ in range(max(1, copies)):
            pipe = ov.WhisperPipeline(str(model_dir), device, CACHE_DIR=str(cache_dir))
            cfg = pipe.get_generation_config()
            if not model_dir.name.split("-int")[0].endswith(".en"):        # English-only models refuse a language or task token
                cfg.language = f"<|{language}|>"
                cfg.task = "transcribe"
            if hotwords and hasattr(cfg, "initial_prompt"):
                cfg.initial_prompt = ", ".join(hotwords[:60])
            self._free.put((pipe, cfg))
        self.copies = max(1, copies)

    def warm_up(self) -> None:
        held = [self._free.get() for _ in range(self.copies)]                       # every copy compiles / loads once
        try:
            for pipe, cfg in held:
                pipe.generate(np.zeros(SAMPLE_RATE, dtype=np.float32).tolist(), cfg)
        finally:
            for h in held:
                self._free.put(h)

    def transcribe_float(self, audio: np.ndarray, prompt: str | None = None) -> str:
        pipe, cfg = self._free.get()                                              # a pipeline is not re-entrant: each turn takes a free copy
        try:
            if prompt is not None and hasattr(cfg, "initial_prompt"):
                cfg.initial_prompt = prompt                                       # what to expect right now (see voice/prompts.py)
            res = pipe.generate(audio.astype(np.float32).tolist(), cfg)
        finally:
            self._free.put((pipe, cfg))
        return str(res.texts[0] if hasattr(res, "texts") else res).strip()

    def transcribe(self, pcm16: bytes, prompt: str | None = None) -> tuple[str, float]:
        """16 kHz mono signed 16-bit PCM -> (text, milliseconds)."""
        t0 = time.perf_counter()
        text = self.transcribe_float(np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0, prompt)
        return text, (time.perf_counter() - t0) * 1000


def wav_bytes(pcm16: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm16)
    return buf.getvalue()


class RemoteWhisper:
    """Recognition on a shared server with the OpenAI transcription API (`POST {url}/audio/transcriptions`): vLLM serving a Whisper model,
    speaches / faster-whisper-server, or any compatible server on the company's GPUs.

    Why: in the worker, Whisper takes CPU time from every call's speech detection and voice, and one copy serves one turn at a time. On a GPU
    server the turns of all workers are batched (large-v3-turbo there is faster than base on a laptop CPU), and the workers carry more calls.
    Same interface as WhisperOpenVino. A server error or timeout is "nothing heard": the caller is asked to say it again, the call goes on."""

    def __init__(self, url: str, model: str, timeout_s: float = 3.0, hotwords: list[str] | None = None, language: str = "en",
                 max_parallel: int = 64, client=None) -> None:
        import httpx

        self.base = url.rstrip("/")
        self.model, self.timeout_s, self.language = model, timeout_s, language
        self.default_prompt = ", ".join((hotwords or [])[:60]) or None
        self.name = f"remote-whisper:{model}@{self.base}"
        self.client = client or httpx.Client(timeout=timeout_s, limits=httpx.Limits(max_connections=max_parallel, max_keepalive_connections=max_parallel))
        self.errors = 0

    def warm_up(self) -> None:
        """Fails start-up loudly if the server is not there (a worker that cannot hear must not take calls)."""
        self._post(wav_bytes(b"\x00\x00" * (SAMPLE_RATE // 2)), None, raise_errors=True)

    def _post(self, wav: bytes, prompt: str | None, raise_errors: bool = False) -> str:
        data = {"model": self.model, "language": self.language, "response_format": "json", "temperature": "0"}
        if prompt or self.default_prompt:
            data["prompt"] = prompt or self.default_prompt
        try:
            r = self.client.post(f"{self.base}/audio/transcriptions", data=data, files={"file": ("turn.wav", wav, "audio/wav")}, timeout=self.timeout_s)
            r.raise_for_status()
            return str(r.json().get("text", "")).strip()
        except Exception:
            self.errors += 1
            if raise_errors:
                raise
            return ""

    def transcribe_float(self, audio: np.ndarray, prompt: str | None = None) -> str:
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        return self._post(wav_bytes(pcm), prompt)

    def transcribe(self, pcm16: bytes, prompt: str | None = None) -> tuple[str, float]:
        t0 = time.perf_counter()
        text = self._post(wav_bytes(pcm16), prompt)
        return text, (time.perf_counter() - t0) * 1000
