r"""Runs every line in tests/data/real_utterances.yaml through the real understanding pipeline (rules, then the real model) and reports misses.

  .venv\Scripts\python.exe scripts\replay_utterances.py [--backend openvino_gpu] [--times 3]

--times N asks the model N times per line: the model's answers on short replies vary, and a line that is right only sometimes is a bug.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="openvino_gpu")
    ap.add_argument("--times", type=int, default=3)
    a = ap.parse_args()

    from dealer_core.app import build_app
    from dealer_core.config import load_config
    from dealer_core.dialogue.states import Intent, State

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=Path(r"C:\dev\bmw-portal\site\data.js"), backend=a.backend, reply_mode="template")
    print("loading the model...", flush=True)
    app.chain.start()
    cases = yaml.safe_load((ROOT / "tests" / "data" / "real_utterances.yaml").read_text(encoding="utf-8"))
    bad = 0
    for c in cases:
        results = []
        for _ in range(a.times if c["via"] == "any" else 1):
            u = app.engine.nlu.understand(State(c["state"]), c.get("car"), c["text"], c["last_agent"])
            ok = (u.intent is Intent(c["intent"])) if "intent" in c else (u.intent is not Intent(c["intent_not"]))
            results.append((ok, u.intent.value, u.via))
        good = sum(r[0] for r in results)
        flag = "ok  " if good == len(results) else "MISS"
        bad += good != len(results)
        print(f"{flag} {c['name']:28s} {good}/{len(results)}  got {sorted({r[1] + '/' + r[2] for r in results})}  expected {c.get('intent') or 'not ' + c['intent_not']}")
    print(f"\n{len(cases) - bad}/{len(cases)} lines understood every time")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
