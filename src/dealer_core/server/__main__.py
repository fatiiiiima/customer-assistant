"""Start the website's voice server:  python -m dealer_core.server --site <the website's site folder>"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Voice server for the AGMC website (all local).")
    ap.add_argument("--site", type=Path, default=Path(r"C:\dev\bmw-portal\site"), help="the website's site folder (its data.js is the inventory)")
    ap.add_argument("--host", default="127.0.0.1", help="interface to listen on; keep 127.0.0.1 unless the page is served from another machine")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-calls", type=int, default=2, help="simultaneous calls; more are told 'busy' (the page falls back to text chat)")
    ap.add_argument("--origin", action="append", help="a page origin allowed to connect (repeatable); default http://localhost:8080")
    ap.add_argument("--backend", default="openvino_gpu", help="llm backend from config; the voice path uses openvino_gpu (the iGPU)")
    ap.add_argument("--mute-while-speaking", action="store_true", help="ignore the microphone while the agent talks (if the agent hears itself on laptop speakers; it can clip a fast reply, so it is off by default and the browser cancels echo)")
    ap.add_argument("--agent", action="store_true", help="tool-calling agent engine: the model picks the next action, code runs it (experimental; default is the state machine)")
    ap.add_argument("--fixed-silence", action="store_true", help="end turns after a fixed silence instead of Smart Turn")
    ap.add_argument("--mode", choices=["conversational", "hybrid", "template"], help="who writes the reply; default dialogue.reply_mode")
    ap.add_argument("--stt-model", help="OpenVINO Whisper folder under models/ov (default voice.stt_model)")
    a = ap.parse_args()

    import uvicorn

    from ..app import ROOT, build_app
    from ..cli.chat import start
    from ..config import load_config
    from ..voice.pipeline import apply_voice_profile, load_engines
    from .app import ServerState, create_server

    data_js = a.site / "data.js"
    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=data_js, backend=a.backend, reply_mode=a.mode, engine="agent" if a.agent else None)
    start(app)                                                     # loads and warms the language model
    apply_voice_profile(app, cfg)
    print("Loading speech models...", flush=True)
    parts = load_engines(cfg, a.stt_model)
    smart = cfg.voice.smart_turn and not a.fixed_silence
    state = ServerState(app, parts, cfg, max_calls=a.max_calls, stop_secs=cfg.voice.smart_stop_secs if smart else cfg.voice.stop_secs,
                        smart_turn=smart, smart_fallback_secs=cfg.voice.smart_fallback_secs, half_duplex=a.mute_while_speaking)
    origins = a.origin or ["http://localhost:8080", "http://127.0.0.1:8080"]
    print(f"Voice server ready on ws://{a.host}:{a.port}/voice  (up to {a.max_calls} calls; pages allowed: {', '.join(origins)})", flush=True)
    uvicorn.run(create_server(state, origins), host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
