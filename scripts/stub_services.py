r"""Stand-ins for the two shared services that limit this Mac, so a load test can ask about everything else (the router, the workers, the database, the WebSocket door).

  .venv/bin/python scripts/stub_services.py --port 9100
  VA__VOICE__STT_URL=http://127.0.0.1:9100/v1  VA__TTS__POCKET_URL=http://127.0.0.1:9100/v1  VA__TTS__POCKET_FALLBACK_URL=http://127.0.0.1:9100/v1

* `POST /v1/audio/transcriptions` (the recogniser): answers a phrase from a short list, after a delay (default 130 ms, the measured Qwen3 time for a turn). The audio is not listened to.
* `POST /v1/audio/speech` (the voice, `stream: true`, `response_format: pcm`): streams a quiet 220 Hz tone as 24 kHz 16-bit audio, as long as the text would take to say (about 15
  characters a second), the first piece after 35 ms and the rest at 7.8 times real time: the measured Pocket-TTS speeds (ARCH_REVIEW section 101).
Neither takes one request at a time (the real ones do), so a test with these shows what the rest of the system does, not what the services would. Nothing here is logged.
"""
from __future__ import annotations

import argparse
import asyncio
import random

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

RATE = 24000
PHRASES = ["Do you have any Toyota", "Tell me more about the first one", "What is the price", "How many kilometres has it done", "Can I come and see it on Saturday",
           "Ten in the morning is fine", "My name is Omar Hassan", "Yes that is right", "What time are you open today", "Thank you that is all"]
CHARS_PER_SECOND = 15.0
FIRST_CHUNK_S = 0.035
REAL_TIME_FACTOR = 7.8
PIECE_S = 0.1                                           # audio per piece sent


def tone(seconds: float) -> bytes:
    t = np.arange(int(seconds * RATE)) / RATE
    return (np.sin(2 * np.pi * 220 * t) * 0.05 * 32767).astype("<i2").tobytes()


def make_app(stt_delay_s: float = 0.13, speed: float = REAL_TIME_FACTOR) -> FastAPI:
    app = FastAPI()
    count = {"stt": 0, "tts": 0}

    @app.get("/health")
    @app.get("/v1/models")
    async def health():
        return {"ok": True, "data": [{"id": "stub", "object": "model"}], **count}

    @app.post("/v1/audio/transcriptions")
    async def transcribe(request: Request):
        await request.body()                                                 # take the upload, ignore the audio
        await asyncio.sleep(stt_delay_s * random.uniform(0.8, 1.2))
        count["stt"] += 1
        return JSONResponse({"text": PHRASES[count["stt"] % len(PHRASES)], "ms": round(stt_delay_s * 1000)})

    @app.post("/v1/audio/speech")
    async def speak(request: Request):
        body = await request.json()
        count["tts"] += 1
        seconds = max(0.3, len(str(body.get("input", ""))) / CHARS_PER_SECOND)

        async def pieces():
            await asyncio.sleep(FIRST_CHUNK_S)
            left = seconds
            while left > 0:
                n = min(PIECE_S, left)
                yield tone(n)
                left -= n
                await asyncio.sleep(n / speed)
        return StreamingResponse(pieces(), media_type="audio/pcm")
    return app


def main() -> int:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--stt-delay", type=float, default=0.13, help="seconds for one recognition")
    a = ap.parse_args()
    uvicorn.run(make_app(a.stt_delay), host=a.host, port=a.port, log_level="warning", access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
