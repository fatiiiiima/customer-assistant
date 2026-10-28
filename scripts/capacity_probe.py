r"""How many simultaneous calls can THIS machine carry, at which quality level, and how many such machines are needed for N callers?

  .venv\Scripts\python.exe scripts\capacity_probe.py                       # the model, recognizer and voice this config uses
  .venv\Scripts\python.exe scripts\capacity_probe.py --backend gpu_server --target 500

Run it on whatever machine the company provides (or against its model server). Nothing about the hardware is assumed: everything is measured.

It measures, with real prompts from this agent:
  model    how many replies can be written at the same time while the first sentence still arrives within the budget (full quality),
           and how many short follow-up questions (prefix quality) within the same budget
  speech   recognitions per second on the configured device
  voice    sentences synthesised per second

and turns that into callers, from how often a caller actually needs each part: in our recorded test calls a caller takes a turn about every
`--turn-every` seconds (default 15: they listen to the reply, think, speak), so one model slot serves several callers.
The estimate is an estimate: confirm it with scripts/load_test.py against the real deployment before going live.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import re
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_PORTAL_DATA = ROOT.parent / "bmw-portal" / "site" / "data.js"     # was C:\dev\bmw-portal\...: the same file on the Windows layout, and found on any other machine

CALLER_LINES = ["is the RAV4 a good family car", "how does it compare with the Camry", "what would you recommend for a family of five",
                "tell me about the Land Cruiser", "is it reliable", "which one is the fastest", "anything cheaper than that", "does it have a sunroof"]


def first_sentence_ms(backend, system: str, user: str, max_tokens: int, prefill: str = "") -> float:
    t0 = time.perf_counter()
    buf = ""
    for piece in backend.lead_in_stream(system, user, max_tokens, 0.7, 20.0, prefill=prefill) if prefill else backend.lead_in_stream(system, user, max_tokens, 0.7, 20.0):
        buf += piece
        if re.search(r"[.!?](?:\s|$)", buf):
            break
    return (time.perf_counter() - t0) * 1000


def sweep(label: str, fn, levels: list[int], budget_ms: float, stop_early: bool = True) -> tuple[int, list[dict]]:
    """Run `fn` N at a time; the largest N whose 90th percentile stays within the budget. Stops once a level is over twice the budget, unless `stop_early` is off."""
    rows, best = [], 0
    for n in levels:
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(n) as pool:
            times = list(pool.map(lambda i: fn(i), range(n * 2)))           # two rounds, so every slot is busy the whole time
        per_s = len(times) / (time.perf_counter() - t0)                     # requests the server finished per second at this many at once
        p90 = float(np.percentile(times, 90))
        rows.append({"at_once": n, "median_ms": round(st.median(times)), "p90_ms": round(p90), "requests_per_s": round(per_s, 1)})
        print(f"  {label:7s} {n:3d} at once: first sentence median {st.median(times):6.0f} ms, p90 {p90:6.0f} ms, {per_s:5.1f} requests/s", flush=True)
        if p90 <= budget_ms:
            best = n
        elif stop_early and p90 > budget_ms * 2:
            break
    return best, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="openvino_gpu")
    ap.add_argument("--budget-ms", type=float, default=None, help="time to the model's first sentence that keeps a reply under ~1 s (default latency.model_budget_ms)")
    ap.add_argument("--target", type=int, default=500, help="simultaneous callers to plan for")
    ap.add_argument("--turn-every", type=float, default=15.0, help="seconds between one caller's turns (measured in our test calls: ~15)")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--no-stop", action="store_true", help="run every level, even after one is over twice the budget (the gate is measured at 50-100 at once, which a slow card would never reach)")
    ap.add_argument("--model-only", action="store_true", help="measure only the model server: skip this machine's own speech recognition and voice timing (they say nothing about a GPU server)")
    ap.add_argument("--portal-data", type=Path, default=DEFAULT_PORTAL_DATA, help="the website's site/data.js (default: ../bmw-portal/site/data.js beside this repository)")
    a = ap.parse_args()

    from dealer_core.app import build_app
    from dealer_core.cli.chat import start
    from dealer_core.config import load_config
    from dealer_core.dialogue.writer import SYSTEM
    from dealer_core.agent.engine import CONTINUE_SYSTEM
    from dealer_core.voice.pipeline import load_engines

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    budget = a.budget_ms or cfg.latency.model_budget_ms
    app = build_app(cfg, inventory="portal", portal_data=a.portal_data, engine="agent", backend=a.backend)
    start(app, quiet=True)
    backend = app.engine.reply_writer.backend
    facts = ("- 2021 Toyota RAV4: white, 98,500 dirhams, 5 seats, all-wheel drive, hybrid, 219 hp, uses 4.9 L/100 km, boot 580 litres, accident-free\n"
             "- 2020 Toyota Camry: brown, 74,500 dirhams, 5 seats, front-wheel drive, hybrid, 215 hp, uses 4.5 L/100 km, boot 524 litres")

    def full(i):
        user = (f"CONVERSATION SO FAR:\nAGENT: There's the Corolla, the Camry and the RAV4.\n\nCUSTOMER (just now): {CALLER_LINES[i % len(CALLER_LINES)]}\n\n"
                f"FACTS FROM THE SYSTEM:\n{facts}\n\nSTANDARD WORDING (reference only, do not copy): The RAV4 seats five. Want to book a test drive?\n\n"
                "This is spoken on the phone: keep it under 30 words.")
        return first_sentence_ms(backend, SYSTEM, user, 110)

    def prefix(i):
        user = (f"CUSTOMER (just now): {CALLER_LINES[i % len(CALLER_LINES)]}\n\nYOUR REPLY HAS ALREADY STARTED, you have just said: \"The RAV4 seats five.\"\n"
                "Continue your reply with ONE short sentence (at most 14 words) that asks it in your own words.")
        return first_sentence_ms(backend, CONTINUE_SYSTEM, user, 36, prefill="The RAV4 seats five. ")

    print(f"model: {a.backend} ({cfg.llm.backends[a.backend].model}); budget for the first sentence {budget:.0f} ms", flush=True)
    full(0), prefix(0)                                                        # warm up
    full_n, full_rows = sweep("full", full, a.levels, budget, not a.no_stop)
    prefix_n, prefix_rows = sweep("prefix", prefix, a.levels, budget + 2000, not a.no_stop)   # code speaks the facts first (~2 s): the question only has to be ready by then

    if a.model_only:
        out = {"backend": a.backend, "budget_ms": budget, "full": full_rows, "prefix": prefix_rows}
        (ROOT / "logs").mkdir(exist_ok=True)               # logs/ is not in a fresh clone
        (ROOT / "logs" / "capacity_probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        return 0
    parts = load_engines(cfg)
    clip = (np.random.default_rng(1).normal(0, 0.05, 16000 * 3)).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(6):
        parts.stt_engine.transcribe_float(clip, "")
    stt_per_s = 6 / (time.perf_counter() - t0)
    t0 = time.perf_counter()
    sentences = ["It's the 2021 RAV4 in white, with 33,000 kilometres on it."] * 8
    with cf.ThreadPoolExecutor(4) as pool:
        list(pool.map(parts.tts_engine.synthesize, sentences))
    tts_per_s = len(sentences) / (time.perf_counter() - t0)

    # a caller needs, per turn: about one model reply slot for ~1.5 s, one recognition, ~3 sentences of voice
    turns_per_s_per_caller = 1.0 / a.turn_every
    model_s = 1.5
    callers = {
        "full quality (model capacity)": full_n / (model_s * turns_per_s_per_caller) if full_n else 0,
        "prefix quality (model capacity)": prefix_n / (0.6 * turns_per_s_per_caller) if prefix_n else 0,
        "speech recognition": stt_per_s / turns_per_s_per_caller,
        "voice synthesis": tts_per_s / (3 * turns_per_s_per_caller),
        "voice-server processes (measured ~20 calls each)": 20.0,
    }
    full_cap = min(callers["full quality (model capacity)"], callers["speech recognition"], callers["voice synthesis"])
    prefix_cap = min(callers["prefix quality (model capacity)"], callers["speech recognition"], callers["voice synthesis"])
    print("\nWhat this machine can carry (callers at the same time):")
    for k, v in callers.items():
        print(f"  {k:52s} ~{v:6.0f}")
    print(f"\n  => at full quality: ~{full_cap:.0f} callers per machine; at prefix quality: ~{prefix_cap:.0f}")
    print(f"     voice-server processes needed per machine: {math.ceil(max(full_cap, prefix_cap) / 20) or 1} (each ~20 calls)")
    for label, cap in (("full quality", full_cap), ("prefix quality", prefix_cap)):
        print(f"  For {a.target} callers at {label}: {math.ceil(a.target / cap) if cap else 'n/a'} machines like this one" + (" (plus one spare)" if cap else ""))
    out = {"backend": a.backend, "budget_ms": budget, "full": full_rows, "prefix": prefix_rows, "stt_per_s": stt_per_s, "tts_sentences_per_s": tts_per_s,
           "callers": callers, "full_cap": full_cap, "prefix_cap": prefix_cap, "target": a.target}
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "capacity_probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
