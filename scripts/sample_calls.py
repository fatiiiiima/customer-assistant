r"""Whole conversations through the agent with the real model, printed turn by turn, to read what callers would hear (docs/ARCH_REVIEW.md section 30b).

  uv run python scripts/sample_calls.py                         # the three calls, as configured (fillers off), model at gpu_server
  uv run python scripts/sample_calls.py --openers               # with the fillers ("Okay,", "So,") as before 29 Oct, to compare

Text only: no audio, no recogniser. The model writes at the full level with `model_speaks` on and no first-sentence deadline, as the owner's worker runs. The time in
brackets is when the first sentence was ready after the caller's words (the voice adds about 0.4 s); 0.00 s is the code answering by itself. The clock is fixed
(Tuesday 27 Oct 2026, 11:00 Dubai). The model server must be up: the run stops if it cannot be reached rather than print another model's replies.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

CALLS = {
    "The owner's 28 Oct test call (BMW X5)": None,                 # replay_model_speaks.TEST_CALL
    "Family SUV, haggling, booking": ["Hi, I'm looking for a family SUV under 100,000.", "Tell me about the RAV4.", "Is it a good family car?",
                                      "What's the fuel economy like?", "Can you do 90,000?", "Come on, can you do 90?", "Tell me about the first one.",
                                      "Does it have any accidents?", "Okay, I'd like to test drive it.", "Tomorrow afternoon.", "2 PM.", "Ahmed, 050 123 4567", "Yes, book it."],
    "Porsche questions": ["Do you have any Porsches?", "Yes, tell me about the Macan.", "Tell me more about the engine.", "And the boot?", "How many owners has it had?",
                          "What colour is it?", "Thanks, that's all."],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="gpu_server")
    ap.add_argument("--openers", action="store_true", help="say the fillers as before 29 Oct, to compare")
    ap.add_argument("--portal-data", type=Path, default=ROOT.parent / "bmw-portal" / "site" / "data.js")
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "sample_calls.json")
    a = ap.parse_args()

    from replay_model_speaks import NOW, TEST_CALL

    from dealer_core.app import build_app
    from dealer_core.cli.chat import start
    from dealer_core.config import load_config
    from dealer_core.voice.pipeline import apply_voice_profile

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=a.portal_data, engine="agent", backend=a.backend, quality="full", clock=lambda: NOW.astimezone(timezone.utc))
    start(app, quiet=True)
    apply_voice_profile(app, cfg)
    e = app.engine
    writer = e.reply_writer.backend
    health = [b.health() for b in (getattr(writer, "backends", None) or [writer]) if getattr(b, "name", "") == a.backend and hasattr(b, "health")]
    if not health or not health[0].ok:
        print(f"the {a.backend} model server cannot be reached ({cfg.llm.backends[a.backend].host}): start it first")
        return 1
    e.model_speaks = True
    e.openers = a.openers or e.openers
    e.reply_writer.first_deadline_s = None
    print(f"model {cfg.llm.backends[a.backend].model}; fillers {'on' if e.openers else 'off'}\n")
    out = {}
    for title, lines in CALLS.items():
        s, turns = e.new_session(), []
        print(f"=== {title}")
        for text in lines or TEST_CALL:
            t0, parts, first = time.perf_counter(), [], None
            for x in e.handle_stream(s, text):
                if isinstance(x, str):
                    parts.append(x)
                    first = first if first is not None else time.perf_counter() - t0
            turns.append({"caller": text, "agent": " ".join(parts), "first_s": round(first or 0.0, 2), "level": e.last_level})
            print(f"CALLER: {text}\n  AGENT [{first or 0:.2f}s]: {' '.join(parts)}")
        out[title] = turns
        print()
    a.out.parent.mkdir(exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
