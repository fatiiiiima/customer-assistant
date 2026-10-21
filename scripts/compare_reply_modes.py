r"""Timing and fallback comparison of the three reply modes on the same invented conversations (NOT a tuned or held-out set).

  .venv\Scripts\python.exe scripts\compare_reply_modes.py [--backend openvino_gpu]
Prints, per mode: turn-time percentiles for the turns a mode can change (search / car question / off-topic), the share of those turns
where the model's text was actually used, and every spoken reply side by side.
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dealer_core.app import build_app, demo_clock  # noqa: E402
from dealer_core.config import load_config  # noqa: E402
from dealer_core.nlu.chain import BackendChain  # noqa: E402

CALLS = [
    ["hello", "i'm looking for an suv", "toyota please", "i want another car instead", "a sedan under 90 thousand", "is it red",
     "do you have a bently", "how many kilometres has it done"],
    ["hi there", "do you have a land cruiser", "is it automatic", "is it white", "is there only one", "what's the weather like today",
     "i like football", "any budget"],
    ["good morning", "i need a family car", "something with low mileage", "show me something different", "the cheapest one"],
    ["hey", "a pickup please", "do you have anything else", "do you have a camry", "is it red", "what colour is it", "thanks bye"],
]
ELIGIBLE = {"search_cars", "ask_car_info", "out_of_scope"}


def run(cfg, chain, **flags):
    app = build_app(cfg, chain=chain, clock=demo_clock(cfg), **flags)
    rows = []
    for turns in CALLS:
        s = app.engine.new_session()
        app.engine.opening(s)
        for t in turns:
            r = app.engine.handle(s, t)
            d = r.debug
            rows.append(dict(user=t, reply=r.reply, intent=d.intent, ms=d.total_ms, used=bool(d.lead_in or d.draft),
                             why=d.note.split("dropped (")[-1].rstrip(")") if "dropped" in d.note else "", via=d.via.split(":")[0]))
    return rows


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="openvino_gpu")
    ap.add_argument("--modes", default="template,lead-in,draft")
    a = ap.parse_args()
    cfg = load_config()
    chain = BackendChain.from_config(cfg, primary=a.backend)
    print("loading model...", flush=True)
    chain.start()
    modes = {"template": dict(lead_in=False, draft=False), "lead-in": dict(lead_in=True, draft=False), "draft": dict(lead_in=False, draft=True), "mixed": dict(lead_in=True, draft=True)}
    results = {m: run(cfg, chain, **f) for m, f in modes.items() if m in a.modes.split(",")}
    print("\n" + "=" * 100 + "\nTURN TIME (ms) for turns the mode can change (search / car question / off-topic)")
    print(f"{'mode':10s} {'turns':>5s} {'p50':>7s} {'p95':>7s} {'max':>7s} {'model text used':>16s}")
    for m, rows in results.items():
        el = [r for r in rows if r["intent"] in ELIGIBLE]
        ms = [r["ms"] for r in el]
        used = sum(r["used"] for r in el)
        print(f"{m:10s} {len(el):5d} {st.median(ms):7.0f} {pct(ms, 95):7.0f} {max(ms):7.0f} {used:8d} / {len(el)} ({100 * used / len(el):.0f}%)")
    print("\nTURN TIME (ms), ALL turns: " + ", ".join(f"{m} p50 {st.median([r['ms'] for r in rows]):.0f} / p95 {pct([r['ms'] for r in rows], 95):.0f}"
                                                   for m, rows in results.items()))
    drops = {}
    for r in results.get("draft", []) + results.get("mixed", []):
        if r["why"] and r["intent"] in ELIGIBLE:
            for part in r["why"].split("; "):
                key = part.split("(")[0].split("[")[0].strip()
                drops[key] = drops.get(key, 0) + 1
    print("draft rejections by reason:", drops or "none")
    for r in results.get("draft", []):
        if r["why"] and r["intent"] in ELIGIBLE:
            print(f"   rejected [{r['user']}]: {r['why']}")
    print("\n" + "=" * 100 + "\nREPLIES SIDE BY SIDE (eligible turns)")
    for i, base in enumerate(results["template"]):
        if base["intent"] not in ELIGIBLE:
            continue
        print(f"\nYOU: {base['user']}   [{base['intent']} via {base['via']}]")
        for m in results:
            r = results[m][i]
            tag = "model" if r["used"] else "template"
            print(f"  {m:9s} ({r['ms']:5.0f} ms, {tag:8s}) {r['reply']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
