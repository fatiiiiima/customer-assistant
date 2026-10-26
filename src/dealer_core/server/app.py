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
    calls: object = None                  # shared.state.CallStore: each call saved after every turn, so another worker can carry it on
    registry: object = None               # shared.state.WorkerRegistry: this worker's address and load, for the router
    worker_id: str = ""
    public_url: str = ""                  # the address the router hands to pages for this worker


def create_server(state: ServerState, allowed_origins: list[str] | None = None) -> FastAPI:
    api = FastAPI(title="AGMC voice server")
    origins = allowed_origins or ["http://localhost:8080", "http://127.0.0.1:8080"]
    api.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"], allow_headers=["*"])

    @api.on_event("startup")
    async def _heartbeat() -> None:
        if state.registry is None:
            return

        async def beat() -> None:                             # every 2 s: "I am here, this busy" (expires in 6 s if this worker dies)
            while True:
                gov = getattr(state.app.engine, "governor", None)
                state.registry.beat(state.worker_id, state.public_url, state.active, state.max_calls,
                                    {"quality": gov.snapshot()["turns"] if gov is not None else {}})
                await asyncio.sleep(2.0)

        asyncio.get_running_loop().create_task(beat())

    @api.on_event("shutdown")
    async def _leave() -> None:
        if state.registry is not None:
            state.registry.leave(state.worker_id)

    @api.on_event("startup")
    async def _threads() -> None:
        # Each call keeps a few threads busy while it waits on shared models (the turn, its speech pieces, the recognizer). Python's default
        # pool (cores + 4 = 20 here) ran out at about 20 callers, so turns queued behind other calls' waits: give every call room.
        from concurrent.futures import ThreadPoolExecutor

        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=max(32, state.max_calls * 4), thread_name_prefix="call"))

    @api.get("/health")
    async def health() -> dict:
        gov = getattr(state.app.engine, "governor", None)
        return {"ok": True, "active_calls": state.active, "max_calls": state.max_calls, "served": state.served, "refused": state.refused,
                "uptime_s": round(time.monotonic() - state.started), "quality": gov.snapshot() if gov is not None else None}

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


def site_bookings(hello: dict, app, tz) -> list:
    """The test drives the page says this visitor already has (car, date, time, and its reference, name and phone when sent)."""
    from datetime import datetime

    from ..dialogue.session import KnownBooking

    out = []
    for b in hello.get("bookings") or []:
        try:
            start = datetime.strptime(f"{b['date']} {b['time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
        except (KeyError, TypeError, ValueError):
            continue
        if app.service.get_car(str(b.get("car_id"))) is None:
            continue
        out.append(KnownBooking(str(b["car_id"]), start, str(b.get("name") or ""), str(b.get("phone") or ""), ref=str(b.get("ref") or "")))
    return out


async def run_call(state: ServerState, ws: WebSocket) -> None:
    hello = await _hello(ws)
    app, parts = state.app, state.parts
    tap = (hello.get("mode") or state.default_mode) == "tap"
    tz = ZoneInfo(state.cfg.branches[0].timezone)
    call_id = str(hello.get("call_id") or "")[:64]
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
        for kb in s.cancelled[seen.setdefault("cancelled", 0):]:              # cancelled or moved in this call: the page removes it from its list
            start = kb.start.astimezone(tz)
            emit({"type": "cancelled", "ref": kb.ref, "car_id": kb.stock_id, "date": start.strftime("%Y-%m-%d"), "time": start.strftime("%H:%M")})
        seen["cancelled"] = len(s.cancelled)
        if state.calls is not None and call_id:
            state.calls.save(call_id, s)                         # after every turn: another worker can carry on from here
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
    gate = MicGate(start_open=False) if tap else None          # tap mode: the greeting is spoken first, the visitor taps to reply

    async def turn_over() -> None:                            # tap-to-talk: the server decided your turn is over: close the microphone, tell the page
        gate.close_after_turn()
        await send({"type": "mic", "on": False, "reason": "turn"})

    resumable = bool(state.calls is not None and call_id and state.calls.store.get(f"call:{call_id}"))   # a reconnect: no greeting again
    pipeline, brain = build_pipeline(app, parts, transport.input(), transport.output(), stop_secs=app.service.cfg.voice.tap_stop_secs if tap and state.smart_turn else state.stop_secs,
                                     mic_gate=gate, on_user_stop=turn_over if tap else None,
                                     half_duplex=state.half_duplex, gate_tail_secs=0.2, barge_in=state.barge_in and not tap, smart_turn=state.smart_turn, smart_fallback_secs=state.smart_fallback_secs, probes=probes, on_turn=on_turn, on_heard=on_heard, on_say=on_say,
                                     speak_opening=not resumable)
    brain_ref["brain"] = brain
    resumed = bool(resumable and state.calls.load(call_id, brain.session, app.service))
    if not resumed:
        brain.session.known_bookings = site_bookings(hello, app, tz)     # "what have I booked?", "cancel my test drive": the ones already on the site
    car = hello.get("car")
    if not resumed and isinstance(car, str) and app.service.get_car(car) is not None:           # the visitor is looking at this car: "this one" means it
        brain.session.car = car
        brain.session.state = State.ANSWERING
    worker = PipelineWorker(pipeline, params=PipelineParams(audio_in_sample_rate=STT_RATE, audio_out_sample_rate=parts.tts_engine.sample_rate),
                            idle_timeout_secs=None)

    @transport.event_handler("on_client_disconnected")
    async def _gone(_transport, _ws) -> None:
        await worker.cancel()

    await send({"type": "ready", "sample_rate": parts.tts_engine.sample_rate, "mode": "tap" if tap else "handsfree", "resumed": resumed})
    if resumed:
        await send({"type": "line", "who": "agent", "turn": 0, "text": "Sorry, we got cut off for a moment. I'm back: where were we?"})
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    limit = asyncio.get_running_loop().call_later(state.max_call_secs, lambda: asyncio.create_task(worker.cancel()))
    try:
        await runner.run()
    finally:
        limit.cancel()
