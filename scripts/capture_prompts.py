r"""Records the exact prompts the voice agent sends to its model during a scripted chat, so other models can be compared on THE SAME prompts.

  .venv\Scripts\python.exe scripts\capture_prompts.py logs\prompts.json

Each record: {"kind": "reply" | "decide", "system": ..., "user": ..., "customer": ..., "tool": (decide only: what the current model chose)}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHAT = ["Toyota", "any budget?", "Carolla", "how much is it", "how many kilometres has it done", "can I book a test drive", "tomorrow at four in the afternoon",
        "my name is Sara Khan", "my number is zero five zero one two three four five six seven", "yes that is right", "thank you, bye"]


def main() -> int:
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "logs" / "prompts.json")
    from dealer_core.app import build_app
    from dealer_core.cli.chat import start
    from dealer_core.config import load_config
    from dealer_core.voice.pipeline import apply_voice_profile

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=Path(r"C:\dev\bmw-portal\site\data.js"), backend="openvino_gpu", engine="agent")
    start(app, quiet=True)
    apply_voice_profile(app, cfg)
    engine = app.engine
    records: list[dict] = []
    current = {"customer": ""}

    real_decide, real_stream = engine.backend.lead_in, engine.reply_writer.backend.lead_in_stream

    def decide(system, user, *a, **k):
        raw = real_decide(system, user, *a, **k)
        records.append({"kind": "decide", "system": system, "user": user, "customer": current["customer"], "tool": raw.strip()})
        return raw

    def stream(system, user, *a, **k):
        records.append({"kind": "reply", "system": system, "user": user, "customer": current["customer"],
                        "max_tokens": a[0] if a else 60})
        yield from real_stream(system, user, *a, **k)

    engine.backend.lead_in = decide
    engine.reply_writer.backend.lead_in_stream = stream
    s = engine.new_session()
    engine.opening(s)
    for line in CHAT:
        current["customer"] = line
        t0 = time.perf_counter()
        list(engine.handle_stream(s, line))
        print(f"{line!r}: {time.perf_counter() - t0:.1f} s")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"saved {len(records)} prompts to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
