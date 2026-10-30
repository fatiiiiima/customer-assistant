r"""Pocket-TTS (Kyutai, 100M, CC-BY-4.0) on the Mac's CPU through Kyutai's own `pocket-tts` package, as a streaming voice server with any of its built-in voices
("eve" by default; the speech toolkit's build has only "alba", docs/ARCH_REVIEW.md section 40). Same API as scripts/chatterbox_server.py.

  ~/pocketenv/bin/python scripts/pocket_server.py                           # http://127.0.0.1:8040/v1/audio/speech
  ~/pocketenv/bin/python scripts/pocket_server.py --voice alba --port 8041

POST /v1/audio/speech {"input": "...", "stream": true, "response_format": "pcm"}: 16-bit mono 24 kHz audio in HTTP chunks, as it is made; without "stream", a WAV.
"voice" may name another built-in voice. GET /health. One sentence at a time. Runs in its own environment (~/pocketenv: pocket-tts and PyTorch), not the
project's. Uses kyutai/pocket-tts-without-voice-cloning (not gated; the built-in voices only). Nothing the caller says or hears is logged.
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

RATE = 24000


class Preempted(Exception):
    pass


def pcm16(x) -> bytes:
    return (np.clip(np.asarray(x, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def from_first_sound(x: np.ndarray, keep_ms: int = 30, level: float = 0.02) -> np.ndarray | None:
    """The audio from `keep_ms` before its first sample at `level` or louder; None if it is all quiet. Section 41: "eve" began each sentence with about
    0.4 s of quiet, so its first audio arrived in 35 ms but was heard about 0.4 s later."""
    loud = np.flatnonzero(np.abs(x) >= level)
    return x[max(0, loud[0] - int(keep_ms / 1000 * RATE)):] if len(loud) else None


class Voice:
    def __init__(self, voice: str, max_tokens: int | None = None, trim_lead: bool = True) -> None:
        from pocket_tts import TTSModel

        self.model = TTSModel.load_model(language="english")
        self.default, self.trim_lead = voice, trim_lead
        self.extra = {"max_tokens": max_tokens} if max_tokens else {}  # the package cuts text longer than this into pieces, each with its own pauses
        self.states = {voice: self.model.get_state_for_audio_prompt(voice)}
        self.lock = threading.Lock()
        self.live_waiting = 0                                  # live sentences waiting: a background one gives way to them (as chatterbox_server.py)
        self._count = threading.Lock()

    def state(self, voice: str | None):
        voice = voice if voice and voice != "default" else self.default
        if voice not in self.states:
            self.states[voice] = self.model.get_state_for_audio_prompt(voice)
        return self.states[voice]

    def chunks(self, text: str, voice: str | None = None, background: bool = False):
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
                stop, started = threading.Event(), not self.trim_lead
                try:
                    for chunk in self.model.generate_audio_stream(self.state(voice), text, stop=stop, **self.extra):
                        if background and self.live_waiting:
                            raise Preempted()
                        x = chunk.detach().cpu().numpy().reshape(-1)
                        if not started:
                            x = from_first_sound(x)
                            if x is None:
                                continue
                            started = True
                        yield pcm16(x)
                finally:
                    stop.set()
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
            self._json(200, {"ok": True, "model": "pocket-tts", "voice": voice.default, "sample_rate": RATE} if self.path.rstrip("/") in ("/health", "/v1/health") else {"error": "not found"})

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
            try:
                voice.state(req.get("voice"))
            except Exception as e:
                return self._json(400, {"error": f"unknown voice: {e}"})
            if not req.get("stream"):
                audio = b"".join(voice.chunks(text, req.get("voice")))
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
                for chunk in voice.chunks(text, req.get("voice"), background=req.get("priority") == "low"):
                    if chunk:
                        self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                        self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except Preempted:
                self.close_connection = True
            except (BrokenPipeError, ConnectionResetError):
                pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Pocket-TTS ({voice.default}) ready on http://{host}:{port}/v1", flush=True)
    httpd.serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8040)
    ap.add_argument("--voice", default="eve", help="a built-in voice: alba, anna, eve, jane, mary, vera, ... (see the pocket-tts package)")
    ap.add_argument("--max-tokens", type=int, help="text tokens per piece (the package's default is 50, which cuts a long sentence of prices in two or three)")
    ap.add_argument("--keep-lead", action="store_true", help="send the quiet at the start of each sentence as the model makes it (by default it is cut to 30 ms)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    voice = Voice(a.voice, a.max_tokens, trim_lead=not a.keep_lead)
    for _ in voice.chunks("Warming up the voice."):
        pass
    print(f"loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    serve(voice, a.host, a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
