"""The voice door for the website: one WebSocket per call, one Pipecat pipeline and one dialogue Session per call; the models are shared.

Shared by every call (loaded once): the language model (one lane: requests queue), Whisper, Piper (each behind a lock), the inventory and the bookings.
Private to each call: the Session (history, car in focus, booking draft), the pipeline, the VAD.

A connection cap protects the machine: past `max_calls` a new caller is told "busy" and the socket closes (the page falls back to text chat).
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner

from ..app import App
from ..config import Config
from ..dialogue.states import State
from ..voice.micgate import MicGate
from ..voice.pipeline import STT_RATE, VoiceParts, build_pipeline
from ..voice.status import StatusLine, TurnClock
from .protocol import PortalSerializer


@dataclass
class ServerState:
    app: App
    parts: VoiceParts
    cfg: Config
    max_calls: int = 2
    max_call_secs: float = 900.0
    stop_secs: float = 0.35               # the VAD silence (short when smart_turn is on)
    smart_turn: bool = False
    barge_in: bool = True
    default_mode: str = "tap"             # "tap" (tap to talk, mic closed while the agent answers) | "handsfree" (open mic for the whole call); the page can ask for either
    half_duplex: bool = False             # ignore the microphone while the agent talks (laptop speakers leak into the mic); False for headphones
    smart_fallback_secs: float = 1.0
    active: int = 0
    served: int = 0
    refused: int = 0
    started: float = field(default_factory=time.monotonic)


def create_server(state: ServerState, allowed_origins: list[str] | None = None) -> FastAPI:
    api = FastAPI(title="AGMC voice server")
    origins = allowed_origins or ["http://localhost:8080", "http://127.0.0.1:8080"]
    api.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"], allow_headers=["*"])

    @api.get("/health")
    async def health() -> dict:
        return {"ok": True, "active_calls": state.active, "max_calls": state.max_calls, "served": state.served, "refused": state.refused,
                "uptime_s": round(time.monotonic() - state.started)}

    @api.websocket("/voice")
    async def voice(ws: WebSocket) -> None:
        origin = ws.headers.get("origin", "")
        if origin and origin not in origins:                    # a page from any other site may not use this server
            await ws.close(code=1008)
            return
        await ws.accept()
        if state.active >= state.max_calls:
            state.refused += 1
            await ws.send_text(json.dumps({"type": "busy"}))
            await ws.close(code=1013)
            return
        state.active += 1
        state.served += 1
        try:
            await run_call(state, ws)
        except Exception as e:                                  # never let one call take the server down
            logger.error(f"call failed: {type(e).__name__}: {e}")
        finally:
            state.active -= 1
            try:
                await ws.close()
            except Exception:
                pass

    return api


async def _hello(ws: WebSocket) -> dict:
    """The page's first message (which car page the visitor is on). Optional: a missing or late hello is not an error."""
    try:
        msg = await asyncio.wait_for(ws.receive(), timeout=3.0)
        if msg.get("text"):
            data = json.loads(msg["text"])
            return data if isinstance(data, dict) and data.get("type") == "hello" else {}
    except (asyncio.TimeoutError, ValueError, KeyError):
        pass
    return {}


async def run_call(state: ServerState, ws: WebSocket) -> None:
    hello = await _hello(ws)
    app, parts = state.app, state.parts
    tap = (hello.get("mode") or state.default_mode) == "tap"
    tz = ZoneInfo(state.cfg.branches[0].timezone)
    turn = {"n": 0}
    seen = {"shown": (), "booking": None}

    async def send(msg: dict) -> None:
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            pass

    def emit(msg: dict) -> None:
        asyncio.get_running_loop().create_task(send(msg))

    def on_heard(text: str) -> None:
        turn["n"] += 1
        turn["said"] = False
        emit({"type": "line", "who": "you", "turn": turn["n"], "text": text})

    def on_say(text: str) -> None:
        turn["said"] = True
        emit({"type": "line", "who": "agent", "turn": turn["n"], "text": text})

    brain_ref: dict = {}

    def on_turn(result) -> None:
        if not turn.get("said", True):                           # model_only test mode: show that the model gave nothing usable, instead of a silent gap
            emit({"type": "line", "who": "agent", "turn": turn["n"], "text": "(the model gave no usable reply)"})
        s = brain_ref["brain"].session
        ids = tuple(c.stock_id for c in s.shown) if not s.car else (s.car,)
        if ids and ids != seen["shown"]:
            seen["shown"] = ids
            emit({"type": "cars", "ids": list(ids)})
        b = s.booking
        if b is not None and b.booking_id != seen["booking"]:
            seen["booking"] = b.booking_id
            start = b.start.astimezone(tz)
            emit({"type": "booking", "car_id": b.stock_id, "name": b.customer_name, "phone": b.customer_phone,
                  "date": start.strftime("%Y-%m-%d"), "time": start.strftime("%H:%M")})

    clock = TurnClock(state.stop_secs, lambda total, parts_: emit({"type": "timing", "seconds": round(total, 2)}))
    probes = {"vad": StatusLine(set(), clock=clock, stage="vad"), "stt": StatusLine(set(), clock=clock, stage="stt"),
              "brain": StatusLine(set(), clock=clock, stage="brain")}
    transport = FastAPIWebsocketTransport(ws, FastAPIWebsocketParams(
        audio_in_enabled=True, audio_out_enabled=True, audio_in_sample_rate=STT_RATE, audio_out_sample_rate=parts.tts_engine.sample_rate,
        serializer=PortalSerializer(), session_timeout=None))
    gate = MicGate() if tap else None

    async def turn_over() -> None:                            # tap-to-talk: the server decided your turn is over: close the microphone, tell the page
        gate.close_after_turn()
        await send({"type": "mic", "on": False, "reason": "turn"})

    pipeline, brain = build_pipeline(app, parts, transport.input(), transport.output(), stop_secs=app.service.cfg.voice.tap_stop_secs if tap and state.smart_turn else state.stop_secs, speak_opening=not tap,
                                     mic_gate=gate, on_user_stop=turn_over if tap else None,
                                     half_duplex=state.half_duplex, gate_tail_secs=0.2, barge_in=state.barge_in and not tap, smart_turn=state.smart_turn, smart_fallback_secs=state.smart_fallback_secs, probes=probes, on_turn=on_turn, on_heard=on_heard, on_say=on_say)
    brain_ref["brain"] = brain
    car = hello.get("car")
    if isinstance(car, str) and app.service.get_car(car) is not None:           # the visitor is looking at this car: "this one" means it
        brain.session.car = car
        brain.session.state = State.ANSWERING
    worker = PipelineWorker(pipeline, params=PipelineParams(audio_in_sample_rate=STT_RATE, audio_out_sample_rate=parts.tts_engine.sample_rate),
                            idle_timeout_secs=None)

    @transport.event_handler("on_client_disconnected")
    async def _gone(_transport, _ws) -> None:
        await worker.cancel()

    await send({"type": "ready", "sample_rate": parts.tts_engine.sample_rate, "mode": "tap" if tap else "handsfree"})
    if tap:                                                   # no spoken greeting in tap mode (the mic is open at once): the greeting is shown as text
        await send({"type": "line", "who": "agent", "turn": 0, "text": app.engine.opening(brain.session)})
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    limit = asyncio.get_running_loop().call_later(state.max_call_secs, lambda: asyncio.create_task(worker.cancel()))
    try:
        await runner.run()
    finally:
        limit.cancel()
