r"""One speech recognizer for every voice worker on the machine, behind the OpenAI transcription API that a GPU server (vLLM, speaches, faster-whisper-server) also offers.

  uv run python -m dealer_core.sttserver --port 8002          then the workers use  VA__VOICE__STT_URL=http://127.0.0.1:8002/v1

Why: each worker otherwise holds its own Whisper (memory, start-up time, CPU taken from the call audio). With this, N workers share one recognizer and
nothing else changes: `voice/stt.py: RemoteWhisper` already speaks this API. In production the same setting points at a batched GPU server instead.

Serves POST /v1/audio/transcriptions (multipart: file = 16 kHz mono WAV, prompt = the hint, language, model), GET /v1/models, GET /health. It uses the same
settings as the workers (voice.stt_model, stt_device, stt_precision, stt_threads), so a turn is recognised exactly as it would be in the worker.
The upload is parsed with the standard library (the `python-multipart` package is not needed).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import time
import wave
from email.parser import BytesParser
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[3]
RATE = 16000


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
        if part.get_filename():
            files[name] = data
        else:
            fields[name] = data.decode("utf-8", "replace")
    return fields, files


def read_wav(data: bytes) -> np.ndarray:
    """A WAV file -> float32 mono at 16 kHz (other rates are resampled linearly; the workers send 16 kHz)."""
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is accepted")
        rate, channels, raw = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    if rate != RATE and len(x):
        x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * RATE / rate)), np.arange(len(x)), x).astype(np.float32)
    return x


def create_app(engine, model_name: str = "whisper") -> FastAPI:
    """`engine` has `transcribe_float(audio, prompt) -> str` (voice.stt.WhisperOpenVino)."""
    api = FastAPI(title="local speech recognizer")
    stats = {"requests": 0, "errors": 0, "in_flight": 0, "started": time.time()}

    @api.post("/v1/audio/transcriptions")
    async def transcriptions(request: Request):
        stats["requests"] += 1
        try:
            fields, files = parse_multipart(await request.body(), request.headers.get("content-type", ""))
            if "file" not in files:
                raise ValueError("no audio file in the request")
            audio = read_wav(files["file"])
        except Exception as e:
            stats["errors"] += 1
            return JSONResponse({"error": {"message": f"bad request: {e}"}}, status_code=400)
        stats["in_flight"] += 1
        try:
            text = await asyncio.to_thread(engine.transcribe_float, audio, fields.get("prompt") or None)
        finally:
            stats["in_flight"] -= 1
        return {"text": text}

    @api.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @api.get("/health")
    async def health():
        return {"ok": True, "model": model_name, **stats, "uptime_s": round(time.time() - stats["started"])}

    return api


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--model", help="folder under models/ov (default voice.stt_model)")
    ap.add_argument("--copies", type=int, default=2, help="recognizers loaded, so turns from different workers are recognised in parallel")
    a = ap.parse_args()
    import uvicorn

    from ..app import ROOT as APP_ROOT
    from ..config import load_config
    from ..voice.stt import WhisperOpenVino, hotword_terms

    cfg = load_config(overlay=APP_ROOT / "config" / "portal.yaml")
    name = a.model or cfg.voice.stt_model
    engine = WhisperOpenVino(APP_ROOT / cfg.app.model_dir / "ov" / name, APP_ROOT / cfg.app.model_dir / "ov_cache", cfg.voice.stt_device,
                             hotword_terms(APP_ROOT / cfg.stt.hotwords_file), copies=a.copies, precision=cfg.voice.stt_precision, threads=cfg.voice.stt_threads)
    engine.warm_up()
    print(f"Speech recognizer ready on http://{a.host}:{a.port}/v1  ({engine.name})", flush=True)
    uvicorn.run(create_app(engine, name), host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
