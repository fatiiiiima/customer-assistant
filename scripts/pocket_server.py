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


class PauseCap:
    """Shortens every quiet stretch inside a sentence to at most `max_ms`, as the audio streams (owner, 30 Oct: "can the speech model speak a little faster
    with no pauses?"). Quiet 10 ms frames are held back until it is known whether speech follows; then at most `max_ms` of them are passed on, half from the
    start of the stretch and half from its end, so the fade out of one word and into the next are kept. Speech itself is never held back."""

    def __init__(self, max_ms: int = 120, share: float = 0.05, floor: float = 0.004) -> None:
        self.frame, self.share, self.floor, self.peak = RATE // 100, share, floor, 0.0     # quiet: under `share` of the loudest frame so far (a breath is quiet too)
        self.keep = max_ms // 10
        self.rest = np.zeros(0, dtype=np.float32)       # less than a frame, waiting for the next chunk
        self.quiet: list[np.ndarray] = []               # the quiet frames held back

    def _kept(self) -> list[np.ndarray]:
        q = self.quiet
        return q if len(q) <= self.keep else q[: self.keep // 2] + q[len(q) - (self.keep - self.keep // 2):]

    def feed(self, x: np.ndarray) -> np.ndarray:
        x = np.concatenate([self.rest, np.asarray(x, dtype=np.float32).reshape(-1)])
        n = len(x) // self.frame * self.frame
        self.rest = x[n:]
        out = []
        for f in x[:n].reshape(-1, self.frame) if n else []:
            lvl = float(np.sqrt(np.mean(f * f)))
            self.peak = max(self.peak, lvl)
            if lvl < max(self.floor, self.share * self.peak):
                self.quiet.append(f)
            else:
                out += self._kept() + [f]
                self.quiet = []
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        """The end of the sentence: the quiet after the last word is kept up to `max_ms` too (the gap before the next sentence)."""
        out = self.quiet[: self.keep] + [self.rest]
        self.quiet, self.rest = [], np.zeros(0, dtype=np.float32)
        return np.concatenate(out)


class Faster:
    """Speaks `speed` times faster at the same pitch (WSOLA: 40 ms windows taken from the input `speed` times further apart than they are laid down, each
    moved by up to 10 ms to where it best continues the last one), as the audio streams. Pocket-TTS has no speed setting of its own."""

    def __init__(self, speed: float = 1.1, frame_ms: int = 40, search_ms: int = 10) -> None:
        self.speed = speed
        self.n = RATE * frame_ms // 1000
        self.hs = self.n // 2                                  # laid down every half window
        self.ha = self.hs * speed                              # taken from the input this far apart
        self.delta = RATE * search_ms // 1000
        self.win = np.hanning(self.n + 1)[:-1].astype(np.float32)     # periodic Hann: halves overlapping at 50% add up to 1
        self.buf = np.zeros(self.delta, dtype=np.float32)      # room to search before the first window
        self.pos = float(self.delta)                           # where the next window is due in buf
        self.prev = None                                       # where the last window was actually taken from
        self.tail = np.zeros(self.hs, dtype=np.float32)        # the second half of the last window, still to be added to

    def _step(self) -> np.ndarray | None:
        target = int(round(self.pos))
        need = max(target + self.delta + self.n, (self.prev + self.hs + self.n) if self.prev is not None else 0)
        if need > len(self.buf):
            return None
        if self.prev is None:
            at = target
        else:
            natural = self.buf[self.prev + self.hs: self.prev + self.hs + self.n]
            region = self.buf[target - self.delta: target + self.delta + self.n]
            at = target - self.delta + int(np.argmax(np.correlate(region, natural, "valid")))
        frame = self.buf[at: at + self.n] * self.win
        out = self.tail + frame[: self.hs]
        self.tail, self.prev, self.pos = frame[self.hs:].copy(), at, self.pos + self.ha
        return out

    def _trim(self) -> None:
        cut = min(int(self.pos) - self.delta, self.prev if self.prev is not None else int(self.pos)) - 1
        if cut > 0:
            self.buf, self.pos = self.buf[cut:], self.pos - cut
            self.prev = self.prev - cut if self.prev is not None else None

    def feed(self, x: np.ndarray) -> np.ndarray:
        if self.speed == 1.0:
            return np.asarray(x, dtype=np.float32).reshape(-1)
        self.buf = np.concatenate([self.buf, np.asarray(x, dtype=np.float32).reshape(-1)])
        out = []
        while (y := self._step()) is not None:
            out.append(y)
        self._trim()
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        if self.speed == 1.0:
            return np.zeros(0, dtype=np.float32)
        end = len(self.buf)
        self.buf = np.concatenate([self.buf, np.zeros(self.n * 2 + self.delta * 2, dtype=np.float32)])
        out = []
        while self.pos < end and (y := self._step()) is not None:
            out.append(y)
        out.append(self.tail)
        return np.concatenate(out)


class Leveller:
    """Turns down whatever is louder than `threshold` (50 ms level), by `ratio`, as the audio streams (owner, 30 Oct: "why does she scream SURE and HI").
    Eve starts most sentences loud: MEASURED, the first 0.4 s at 1.2 to 2.5 times the level of the rest ("Hi, welcome ..." 1.35 to 1.66; "Sure!" alone
    the loudest thing she says). The gain drops at once and comes back up gently; each 10 ms frame looks `ahead_ms` ahead, so a loud start is caught
    before it is played, at the cost of `ahead_ms` of delay."""

    def __init__(self, threshold: float = 0.15, ratio: float = 4.0, ahead_ms: int = 20, release_db_per_s: float = 30.0, makeup: float = 1.15) -> None:
        self.frame, self.threshold, self.ratio, self.makeup = RATE // 100, threshold, ratio, makeup     # makeup: everything raised after, so ordinary sentences are as loud as before
        self.ahead = RATE * ahead_ms // 1000
        self.up = 10 ** (release_db_per_s / 20 / 100)                     # the most the gain may rise in one 10 ms frame
        self.buf = np.zeros(0, dtype=np.float32)                           # not yet played: the next frame and what is ahead of it
        self.past = np.zeros(RATE * 30 // 1000, dtype=np.float32)          # the last 30 ms played (part of the 50 ms window)
        self.gain = 1.0

    def _want(self, window: np.ndarray) -> float:
        level = float(np.sqrt(np.mean(window * window))) if len(window) else 0.0
        return 1.0 if level <= self.threshold else (self.threshold + (level - self.threshold) / self.ratio) / level

    def _frames(self, final: bool) -> np.ndarray:
        out = []
        while len(self.buf) >= self.frame + (0 if final else self.ahead) and len(self.buf) > 0:
            f = self.buf[: self.frame]
            want = self._want(np.concatenate([self.past, self.buf[: self.frame + self.ahead]]))
            g = want if want < self.gain else min(want, self.gain * self.up)
            out.append(f * np.linspace(self.gain, g, len(f), dtype=np.float32) * self.makeup)
            self.gain, self.past, self.buf = g, np.concatenate([self.past, f])[-len(self.past):], self.buf[len(f):]
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def feed(self, x: np.ndarray) -> np.ndarray:
        self.buf = np.concatenate([self.buf, np.asarray(x, dtype=np.float32).reshape(-1)])
        return self._frames(final=False)

    def flush(self) -> np.ndarray:
        return self._frames(final=True)


class Voice:
    def __init__(self, voice: str, max_tokens: int | None = None, trim_lead: bool = True, speed: float = 1.0, max_pause_ms: int = 0,
                 level: float = 0.0) -> None:
        from pocket_tts import TTSModel

        self.model = TTSModel.load_model(language="english")
        self.default, self.trim_lead, self.speed, self.max_pause_ms, self.level = voice, trim_lead, speed, max_pause_ms, level
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
                cap = PauseCap(self.max_pause_ms) if self.max_pause_ms else None
                faster = Faster(self.speed)
                lev = Leveller(self.level) if self.level else None
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
                        x = faster.feed(cap.feed(x) if cap else x)
                        x = lev.feed(x) if lev else x
                        if len(x):
                            yield pcm16(x)
                    x = faster.feed(cap.flush()) if cap else np.zeros(0, dtype=np.float32)
                    x = np.concatenate([x, faster.flush()])
                    x = np.concatenate([lev.feed(x), lev.flush()]) if lev else x
                    if len(x):
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


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8040)
    ap.add_argument("--voice", default="eve", help="a built-in voice: alba, anna, eve, jane, mary, vera, ... (see the pocket-tts package)")
    ap.add_argument("--max-tokens", type=int, help="text tokens per piece (the package's default is 50, which cuts a long sentence of prices in two or three)")
    ap.add_argument("--keep-lead", action="store_true", help="send the quiet at the start of each sentence as the model makes it (by default it is cut to 30 ms)")
    ap.add_argument("--speed", type=float, default=1.0, help="speak this many times faster, same pitch (1.0 = as the model makes it). Owner, 30 Oct: 1.1 (section 43), then "
                    "\"a little slower, the way they do on Fish Audio\" (section 50)")
    ap.add_argument("--level", type=float, default=0.15, help="turn down anything louder than this (50 ms level), 4 to 1, so loud sentence starts do not shout (0 = off). Owner, 30 Oct")
    ap.add_argument("--max-pause-ms", type=int, default=250, help="shorten every pause inside a sentence to at most this (0 = as the model makes it). Owner, 30 Oct: no pauses "
                    "(120), then a calmer pace (250 keeps the pause at a comma, still cuts the long ones; section 50)")
    return ap


def main() -> int:
    a = parser().parse_args()
    t0 = time.perf_counter()
    voice = Voice(a.voice, a.max_tokens, trim_lead=not a.keep_lead, speed=a.speed, max_pause_ms=a.max_pause_ms, level=a.level)
    for _ in voice.chunks("Warming up the voice."):
        pass
    print(f"loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    serve(voice, a.host, a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
