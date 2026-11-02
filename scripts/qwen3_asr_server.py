r"""Qwen3-ASR 0.6B on the Mac's GPU (MLX) as a speech-recognition server with the OpenAI transcription API, for the worker's `voice.stt_url`.

  benchmarks/results/asr_shootout/qwen3-venv/bin/python scripts/qwen3_asr_server.py            # http://127.0.0.1:8011/v1/audio/transcriptions
  VA__VOICE__STT_URL=http://127.0.0.1:8011/v1 VA__VOICE__STT_REMOTE_MODEL=qwen3-asr <the worker's usual command>

Runs in a private MLX environment (`uv venv ...; uv pip install mlx-audio soundfile`), not the project's. The weights are `mlx-community/Qwen3-ASR-0.6B-4bit` (713 MB).

The worker's hint (`prompt`, see voice/prompts.py: stt_prompt) goes to the model as its system prompt, the place Qwen3-ASR takes a context. Why this server and not the
`speech-server` of the Swift toolkit: that one ignores `prompt` (its handler reads only file, language, response_format and model), so car names are not helped.
The language is always forced (default English): left to itself the model once answered an English clip in Russian. Not a streaming model: each turn is recognised whole.
Recognition is one at a time (one GPU stream); a second request waits for the first, which takes about 80 ms for a short turn.

Extra form field for experiments: `hotwords` (comma separated), passed to the model as its hot-word list instead of, or as well as, the prompt.
Nothing the caller said is written to the log. docs/ARCH_REVIEW.md sections 79 to 81.
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
import wave
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO = "mlx-community/Qwen3-ASR-0.6B-4bit"
RATE = 16000
LANGUAGES = {"en": "English", "ar": "Arabic", "hi": "Hindi", "fr": "French", "de": "German", "es": "Spanish", "it": "Italian", "pt": "Portuguese", "zh": "Chinese"}


def language_name(code: str | None) -> str:
    """The worker sends "en"; the model wants "English". Unknown or empty gives English (never auto-detect: it once chose Russian)."""
    if not code:
        return "English"
    return LANGUAGES.get(code.strip().lower(), code.strip().title() if len(code.strip()) > 2 else "English")


def parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], dict[str, bytes]]:
    """(text fields, file parts) of a multipart/form-data body."""
    msg = BytesParser().parsebytes(b"Content-Type: " + content_type.encode("latin-1") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    fields: dict[str, str] = {}
    files: dict[str, bytes] = {}
    if not msg.is_multipart():
        return fields, files
    for part in msg.get_payload():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        data = part.get_payload(decode=True) or b""
        if part.get_filename() is not None:
            files[name] = data
        else:
            fields[name] = data.decode("utf-8", "replace")
    return fields, files


def read_wav(data: bytes) -> np.ndarray:
    """16-bit WAV of any rate and channel count as 16 kHz mono float32."""
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("expects a 16-bit WAV")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        rate, channels = w.getframerate(), w.getnchannels()
    if channels > 1:
        x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
    if rate != RATE:
        from scipy.signal import resample_poly

        g = np.gcd(rate, RATE)
        x = resample_poly(x, RATE // g, rate // g).astype(np.float32)
    return x


class Qwen3ASR:
    """The MLX model behind a lock: one recognition at a time."""

    name = "qwen3-asr-0.6b"

    def __init__(self, repo: str = REPO) -> None:
        from mlx_audio.stt import load

        self.model = load(repo)
        self.lock = threading.Lock()

    def transcribe_float(self, audio: np.ndarray, prompt: str | None = None, language: str | None = "en", hotwords: list[str] | None = None) -> str:
        kw: dict = {"language": language_name(language)}
        if prompt:
            kw["system_prompt"] = prompt
        if hotwords:
            kw["hotwords"] = hotwords
        with self.lock:
            return str(self.model.generate(np.asarray(audio, dtype=np.float32), **kw).text).strip()


def make_handler(engine, model_name: str = "qwen3-asr"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):                                             # no transcripts or caller details in the log
            pass

        def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, body: dict) -> None:
            self._send(code, json.dumps(body).encode())

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", "/v1/models"):
                return self._json(200, {"ok": True, "data": [{"id": model_name, "object": "model"}]})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/audio/transcriptions":
                return self._json(404, {"error": "not found"})
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                fields, files = parse_multipart(body, self.headers.get("Content-Type", ""))
                if "file" not in files:
                    return self._json(400, {"error": "no file"})
                audio = read_wav(files["file"])
            except Exception as e:  # noqa: BLE001 - a bad upload is the caller's, not a crash; only the error's type is returned
                return self._json(400, {"error": type(e).__name__})
            hotwords = [w.strip() for w in fields.get("hotwords", "").split(",") if w.strip()]
            try:
                t0 = time.perf_counter()
                text = engine.transcribe_float(audio, fields.get("prompt") or None, fields.get("language") or "en", hotwords or None)
                ms = round((time.perf_counter() - t0) * 1000)
            except Exception as e:  # noqa: BLE001 - whatever the model raised: an error answer with its type, and the server goes on
                return self._json(500, {"error": type(e).__name__})
            if fields.get("response_format") == "text":
                return self._send(200, text.encode(), "text/plain; charset=utf-8")
            self._json(200, {"text": text, "ms": ms})

    return Handler


def serve(engine, host: str, port: int, model_name: str = "qwen3-asr") -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(engine, model_name))
    httpd.daemon_threads = True
    return httpd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--repo", default=REPO, help="the MLX weights on Hugging Face")
    ap.add_argument("--wav", type=Path, help="recognise this file and exit (a check, no server)")
    ap.add_argument("--prompt", help="with --wav: the hint")
    a = ap.parse_args()
    t0 = time.perf_counter()
    engine = Qwen3ASR(a.repo)
    engine.transcribe_float(np.zeros(RATE, dtype=np.float32))                  # compile and warm up, so the first caller does not pay for it
    print(f"loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    if a.wav:
        t1 = time.perf_counter()
        print(engine.transcribe_float(read_wav(a.wav.read_bytes()), a.prompt), f"({(time.perf_counter() - t1) * 1000:.0f} ms)")
        return 0
    print(f"Qwen3-ASR ready on http://{a.host}:{a.port}/v1", flush=True)
    serve(engine, a.host, a.port).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
