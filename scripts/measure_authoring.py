r"""How much of the conversation is written by the model, how long it takes, and how close it is to the template.

  .venv\Scripts\python.exe scripts\measure_authoring.py [--backend openvino_gpu]
Runs the same invented multi-turn conversations as compare_reply_modes.py (not a tuned or held-out set) in conversational mode and prints
per turn: who wrote the reply, copy ratio (1.0 = identical to the standard wording), time, and why a model reply was rejected.
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dealer_core.app import build_app, demo_clock  # noqa: E402
from dealer_core.config import load_config  # noqa: E402
from dealer_core.nlu.chain import BackendChain  # noqa: E402

CALLS = [
    ["hello", "i want a red car", "the nissan", "yes", "is it automatic", "what is the price", "i cant afford that i can do 45000",
     "ok what about 48000", "its so hot today", "can i come see it saturday afternoon", "4 pm is good"],
    ["hi there", "do you have a land cruiser", "is it white", "how many kilometres has it done", "do you have a bentley", "toyota",
     "any budget", "the second one", "can you do 140000", "my final offer is 130000"],
    ["good morning", "i need a family car", "an suv under 150 thousand", "toyota please", "are they red", "either of them", "i like football",
     "show me something different", "thanks bye"],
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="openvino_gpu")
    a = ap.parse_args()
    cfg = load_config()
    chain = BackendChain.from_config(cfg, primary=a.backend)
    print("loading model...", flush=True)
    chain.start()
    app = build_app(cfg, chain=chain, clock=demo_clock(cfg), reply_mode="conversational")
    rows = []
    for turns in CALLS:
        s = app.engine.new_session()
        print("\n" + "=" * 100 + f"\nAGENT: {app.engine.opening(s)}")
        for t in turns:
            r = app.engine.handle(s, t)
            d = r.debug
            who = "MODEL" if d.draft else "template"
            rows.append((who, d.copy_ratio, d.total_ms, d.risk, d.note))
            print(f"\nYOU: {t}\nAGENT: {r.reply}\n   [{d.intent} | {who}{' copy=%.2f' % d.copy_ratio if d.draft else ''} | risk {d.risk or '-'} | {d.total_ms:.0f} ms]"
                  + (f"\n   rejected: {d.note.split('dropped (')[-1][:150]}" if "dropped" in d.note else ""))
    model = [r for r in rows if r[0] == "MODEL"]
    eligible = [r for r in rows if r[3]]               # turns the writer was asked to author
    print("\n" + "=" * 100)
    print(f"turns the model was asked to write: {len(eligible)}   written by the model: {len(model)} ({100 * len(model) / max(1, len(eligible)):.0f}%)")
    if model:
        print(f"copy ratio of the model's replies: median {st.median(r[1] for r in model):.2f}  (1.00 = identical to the template)")
    ms = [r[2] for r in eligible]
    print(f"turn time: median {st.median(ms):.0f} ms, p95 {sorted(ms)[int(0.95 * (len(ms) - 1))]:.0f} ms, max {max(ms):.0f} ms")
    for tier in ("low", "high", "chat"):
        t = [r for r in eligible if r[3] == tier]
        if t:
            print(f"  {tier:5s}: {sum(r[0] == 'MODEL' for r in t)}/{len(t)} written by the model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
