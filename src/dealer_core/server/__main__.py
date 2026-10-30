"""Start the website's voice server:  python -m dealer_core.server --site <the website's site folder>"""
from __future__ import annotations

import argparse
from pathlib import Path


def drain_on_stop(server_class):
    """uvicorn's server with one addition: on the first SIGTERM / SIGINT the worker marks itself as draining (see ServerState.drain) before uvicorn starts its graceful stop."""

    class Draining(server_class):
        def __init__(self, config, state) -> None:
            super().__init__(config)
            self.state = state

        def handle_exit(self, sig, frame) -> None:
            self.state.drain()
            super().handle_exit(sig, frame)

    return Draining


def main() -> int:
    ap = argparse.ArgumentParser(description="Voice server for the AGMC website (all local).")
    ap.add_argument("--site", type=Path, default=Path(r"C:\dev\bmw-portal\site"), help="the website's site folder (its data.js is the inventory)")
    ap.add_argument("--host", default="127.0.0.1", help="interface to listen on; keep 127.0.0.1 unless the page is served from another machine")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-calls", type=int, default=2, help="the most simultaneous calls (with voice.admission the worker takes fewer while its replies are slow); "
                                                                 "more are told 'busy' (the page falls back to text chat)")
    ap.add_argument("--origin", action="append", help="a page origin allowed to connect (repeatable); default http://localhost:8080")
    ap.add_argument("--quality", choices=["auto", "full", "prefix", "code"], default=None,
                    help="agent: auto picks per turn from the model's speed (default); full / prefix / code force one level")
    ap.add_argument("--state-url", default=None, help="shared state for many workers: postgresql://... or redis://... (default: STATE_URL / DATABASE_URL, else none)")
    ap.add_argument("--worker-id", default=None, help="this worker's name in the registry (default host:port)")
    ap.add_argument("--public-url", default=None, help="the address pages use for this worker (default ws://host:port/voice)")
    ap.add_argument("--tts", choices=["piper", "kokoro", "chatterbox", "pocket"], default=None, help="Wazir's voice engine (default tts.backend)")
    ap.add_argument("--voice", default=None, help="voice name for that engine (piper: joe ...; kokoro: michael, adam, george ...)")
    ap.add_argument("--backend", default="openvino_gpu", help="llm backend from config; the voice path uses openvino_gpu (the iGPU)")
    ap.add_argument("--mute-while-speaking", action="store_true", help="ignore the microphone while the agent talks (if the agent hears itself on laptop speakers; it can clip a fast reply, so it is off by default and the browser cancels echo)")
    ap.add_argument("--hands-free", action="store_true", help="default to an open microphone for the whole call (the page can still ask for tap-to-talk); default is tap-to-talk")
    ap.add_argument("--no-barge-in", action="store_true", help="the caller cannot talk over the agent")
    ap.add_argument("--prefix", action="store_true", help="with --agent: code speaks the facts at once, the model only continues with a short follow-up (experimental)")
    ap.add_argument("--agent", action="store_true", help="tool-calling agent engine: the model picks the next action, code runs it (experimental; default is the state machine)")
    ap.add_argument("--fixed-silence", action="store_true", help="end turns after a fixed silence instead of Smart Turn")
    ap.add_argument("--mode", choices=["conversational", "hybrid", "template"], help="who writes the reply; default dialogue.reply_mode")
    ap.add_argument("--stt-model", help="OpenVINO Whisper folder under models/ov (default voice.stt_model)")
    ap.add_argument("--inventory", choices=["auto", "db", "site"], default="auto",
                    help="db: the cars already in the database (DATABASE_URL); site: import the site's data.js (the demo stock) first; "
                         "auto (default): db when the database has cars, else site")
    a = ap.parse_args()

    import uvicorn

    DrainingServer = drain_on_stop(uvicorn.Server)
    from ..app import ROOT, build_app
    from ..cli.chat import start
    from ..config import load_config
    from ..voice.pipeline import apply_voice_profile, load_engines
    from .app import ServerState, create_server

    data_js = a.site / "data.js"
    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    if a.tts:
        cfg.tts.backend = a.tts
    inventory = a.inventory
    if inventory == "auto":                                        # a database with stock in it is never overwritten by the demo file
        import os

        from ..inventory import open_db
        from ..models import SearchFilters

        url = os.environ.get("DATABASE_URL", "") or cfg.app.database_url
        inventory = "db" if url and open_db(url, cfg.app.db_pool_size)[0].count(SearchFilters(include_unavailable=True)) else "site"
    print(f"Stock: {'the database' if inventory == 'db' else data_js}", flush=True)
    app = build_app(cfg, inventory="db" if inventory == "db" else "portal", portal_data=data_js, backend=a.backend, reply_mode=a.mode, engine="agent" if a.agent else None, agent_prefix=True if a.prefix else None,
                    quality=a.quality)
    start(app)                                                     # loads and warms the language model
    apply_voice_profile(app, cfg)
    gov = getattr(app.engine, "governor", None)
    if gov is not None and gov.mode == "auto" and cfg.dialogue.probe_every_s > 0 and hasattr(app.engine, "probe_model"):
        gov.probed = True                                          # the model's speed is measured by probes, never by a caller's turn
        app.engine.probe_model(record=False)                       # the first request after a load also warms the prompt: not a fair measure
        ms = app.engine.probe_model()
        print("Model speed measured before the first call: " + (f"{ms:.0f} ms to a first sentence ({'model-written replies' if ms <= gov.budget_ms else 'code facts + a short model question'} until it changes)" if ms else "not possible (replies start with the code's own words)"), flush=True)
    print("Loading speech models...", flush=True)
    parts = load_engines(cfg, a.stt_model, a.voice)
    smart = cfg.voice.smart_turn and not a.fixed_silence
    state = ServerState(app, parts, cfg, max_calls=a.max_calls, stop_secs=cfg.voice.smart_stop_secs if smart else cfg.voice.stop_secs,
                        smart_turn=smart, smart_fallback_secs=cfg.voice.smart_fallback_secs, default_mode="handsfree" if a.hands_free else "tap", half_duplex=a.mute_while_speaking, barge_in=cfg.voice.barge_in and not a.no_barge_in)
    import os

    state_url = shared_state_url(a.state_url, os.environ, cfg.app.database_url)
    if state_url:
        from ..shared.state import CallStore, WorkerRegistry, open_store

        store = open_store(state_url, pool_size=cfg.app.db_pool_size)
        state.calls, state.registry = CallStore(store, ttl_s=cfg.voice.calls_kept_secs), WorkerRegistry(store)
        state.worker_id = a.worker_id or f"{a.host}:{a.port}"
        state.public_url = a.public_url or f"ws://{a.host}:{a.port}/voice"
    else:                                                     # one worker: calls kept in this process, so a visitor who hangs up and comes back carries on (section 49)
        from ..shared.state import CallStore, MemoryStore

        state.calls = CallStore(MemoryStore(), ttl_s=cfg.voice.calls_kept_secs)
    from .metrics import Metrics

    if cfg.voice.admission:
        from .admission import AdmissionController

        state.admission = AdmissionController(a.max_calls, target_s=cfg.voice.admission_target_ms / 1000)

    state.metrics = Metrics()
    origins = a.origin or ["http://localhost:8080", "http://127.0.0.1:8080"]
    print(f"Voice server ready on ws://{a.host}:{a.port}/voice  (up to {a.max_calls} calls; pages allowed: {', '.join(origins)})", flush=True)
    server = DrainingServer(uvicorn.Config(create_server(state, origins), host=a.host, port=a.port, log_level="warning"), state)
    server.run()
    return 0


def shared_state_url(arg: str, environ, config_url: str) -> str:
    """Where workers share call state: --state-url, STATE_URL, else the database when it is PostgreSQL. A SQLite stock database
    (sqlite:///stock.db) is one machine's file, not a place for workers to share calls: "" (each worker keeps its own calls)."""
    db = environ.get("DATABASE_URL") or config_url or ""
    return arg or environ.get("STATE_URL") or (db if db.startswith(("postgresql://", "postgres://")) else "")


if __name__ == "__main__":
    raise SystemExit(main())
