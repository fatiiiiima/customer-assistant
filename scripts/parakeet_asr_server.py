r"""Parakeet-TDT 0.6B v2 on the Mac's GPU (MLX) as a speech-recognition server with the OpenAI transcription API, for the worker's `voice.stt_url`.

  benchmarks/results/asr_shootout/qwen3-venv/bin/python scripts/parakeet_asr_server.py            # http://127.0.0.1:8012/v1/audio/transcriptions
  VA__VOICE__STT_URL=http://127.0.0.1:8012/v1 VA__VOICE__STT_REMOTE_MODEL=parakeet VA__VOICE__STT_HOTWORDS_FIELD=true <the worker's usual command>

The same private MLX environment as scripts/qwen3_asr_server.py (mlx-audio, soundfile); the weights are `mlx-community/parakeet-tdt-0.6b-v2` (2.47 GB, English, CC-BY-4.0).
The service itself (upload parsing, WAV to 16 kHz mono, health, error answers, nothing of the caller in the log) is that script's: this file only swaps the model.

Parakeet takes no hint and no word list: `prompt`, `hotwords` and `language` are accepted and ignored (it is English only). The worker setting `voice.stt_hotwords_field`
(true) is how the worker is told that: it then sends no `prompt`, and does not listen a second time with another hint (EchoGuard). Whole clip after the end of speech, about
40 ms for a short turn; recognition is one at a time (one GPU stream). docs/ARCH_REVIEW.md sections 85 and 86.
"""
from __future__ import annotations

import argparse
import importlib.util
import threading
import time
from pathlib import Path

import numpy as np

REPO = "mlx-community/parakeet-tdt-0.6b-v2"
RATE = 16000
_spec = importlib.util.spec_from_file_location("qwen3_asr_server", Path(__file__).with_name("qwen3_asr_server.py"))
service = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(service)


class ParakeetASR:
    """The MLX model behind a lock: one recognition at a time."""

    name = "parakeet-tdt-0.6b-v2"

    def __init__(self, repo: str = REPO) -> None:
        from mlx_audio.stt import load

        self.model = load(repo)
        self.lock = threading.Lock()

    def transcribe_float(self, audio: np.ndarray, prompt: str | None = None, language: str | None = "en", hotwords: list[str] | None = None) -> str:
        import mlx.core as mx

        with self.lock:
            return str(self.model.generate(mx.array(np.asarray(audio, dtype=np.float32))).text).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--repo", default=REPO, help="the MLX weights on Hugging Face")
    ap.add_argument("--wav", type=Path, help="recognise this file and exit (a check, no server)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    engine = ParakeetASR(a.repo)
    engine.transcribe_float(np.zeros(RATE, dtype=np.float32))                  # compile and warm up, so the first caller does not pay for it
    print(f"loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    if a.wav:
        t1 = time.perf_counter()
        print(engine.transcribe_float(service.read_wav(a.wav.read_bytes())), f"({(time.perf_counter() - t1) * 1000:.0f} ms)")
        return 0
    print(f"Parakeet ready on http://{a.host}:{a.port}/v1", flush=True)
    service.serve(engine, a.host, a.port, "parakeet").serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
