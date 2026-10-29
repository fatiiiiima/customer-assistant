r"""Chatterbox Turbo (Resemble AI, MIT) on the Mac's GPU through mlx-audio, as a streaming voice server for the worker (voice/remote_tts.py, `tts.backend: chatterbox`).

  ~/mlxenv312/bin/python scripts/chatterbox_server.py                       # http://127.0.0.1:8030/v1/audio/speech
  VA__TTS__BACKEND=chatterbox <the worker's usual command>                  # trimmed Kokoro stays the fallback

POST /v1/audio/speech {"input": "...", "stream": true, "response_format": "pcm"}: 16-bit mono 24 kHz audio in HTTP chunks, as it is made (mlx-audio sends a chunk every
`--interval` seconds of speech; section 38 measured 1.0 s: first audio about 0.45 s, made in about half the time it takes to say; 0.4 s started sooner but only just kept
up with playback). Without "stream", a WAV. GET /health. One sentence at a time on the GPU (it shares it with the language model). Its built-in voice, or a voice
copied from a 5 to 10 s sample (`--voice-sample`). Runs in the MLX environment (mlx-audio), not the project's. Nothing the caller says or hears is logged.
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

RATE = 24000


class Preempted(Exception):
    pass


def pcm16(x) -> bytes:
    return (np.clip(np.asarray(x, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767).astype(np.int16).tobytes()


class Voice:
    def __init__(self, model: str, interval: float, sample: Path | None) -> None:
        from mlx_audio.tts.utils import load_model

        self.model, self.interval = load_model(model), interval
        self.extra = {"ref_audio": str(sample)} if sample else {}
        self.lock = threading.Lock()
        self.live_waiting = 0                                  # live sentences waiting for the GPU: a background one gives way to them
        self._count = threading.Lock()

    def chunks(self, text: str, stream: bool = True, background: bool = False):
        """The audio of `text`, chunk by chunk. A background sentence (made ahead of time, "priority": "low") stops at its next chunk when a live one is
        waiting, and raises Preempted (29 Oct: the worker's start-up phrases held the caller's sentences back, 1.4 s of voice wait instead of 0.45 s)."""
        if not background:
            with self._count:
                self.live_waiting += 1
        held = False
        try:
            with self.lock:
                if not background:
                    with self._count:
                        self.live_waiting -= 1
                    held = True
                for r in self.model.generate(text=text, stream=stream, streaming_interval=self.interval, **self.extra):
                    if background and self.live_waiting:
                        raise Preempted()
                    yield pcm16(r.audio)
        finally:
            if not background and not held:
                with self._count:
                    self.live_waiting -= 1


def serve(voice: Voice, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self._json(200, {"ok": True, "model": "chatterbox-turbo", "sample_rate": RATE} if self.path.rstrip("/") in ("/health", "/v1/health") else {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/audio/speech":
                return self._json(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                text = str(req.get("input") or "").strip()
            except Exception:
                return self._json(400, {"error": "bad JSON"})
            if not text:
                return self._json(400, {"error": "no input"})
            if not req.get("stream"):
                audio = b"".join(voice.chunks(text, stream=False))
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(RATE)
                    w.writeframes(audio)
                raw = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/pcm")
            self.send_header("X-Sample-Rate", str(RATE))
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for chunk in voice.chunks(text, background=req.get("priority") == "low"):
                    if chunk:
                        self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                        self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except Preempted:                                          # given way to a live sentence: closed without the last chunk, so the worker does not keep it
                self.close_connection = True
            except (BrokenPipeError, ConnectionResetError):           # the worker stopped listening (the caller interrupted): stop making it
                pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Chatterbox Turbo ready on http://{host}:{port}/v1", flush=True)
    httpd.serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8030)
    ap.add_argument("--model", default="mlx-community/chatterbox-turbo-8bit")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds of speech per streamed chunk")
    ap.add_argument("--voice-sample", type=Path, help="a 5 to 10 s WAV of the voice to copy (default: the model's own voice)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    voice = Voice(a.model, a.interval, a.voice_sample)
    for _ in voice.chunks("Warming up the voice."):
        pass
    print(f"loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    serve(voice, a.host, a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
