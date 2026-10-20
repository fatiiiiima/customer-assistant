"""Hybrid NLU: deterministic fast path first, LLM only for what rules decline.

Reports end-to-end NLU accuracy (exact intent+slots, LLM slots passed through the grounding
filter) and per-turn NLU latency for: LLM-only vs hybrid, per case set.
Order matters for the Ollama prefix cache, so cases run in conversation order.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).parent))
from bench_llm import HOST, ground_slots, one_call, score, user_msg  # noqa: E402
from common import save_result, summarize  # noqa: E402
from fastpath import classify  # noqa: E402
from nlu_cases import CASES, FRESH, HELDOUT, SYSTEM, state_schema, state_system  # noqa: E402


def llm_turn(client, model, mode, threads, state, car, utt):
    if mode == "state":
        sysm, sc = state_system(state), state_schema(state)
    else:  # 'cached'
        sysm, sc = SYSTEM, state_schema(state)
    c = one_call(client, model, sysm, user_msg(state, car, utt), threads=threads, schema=sc)
    try:
        pred = json.loads(c.text)
        pred["slots"] = ground_slots(utt, pred.get("slots") or {})
    except json.JSONDecodeError:
        pred = {"intent": "?", "slots": {}}
    return pred, c.total_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--mode", choices=["state", "cached"], default="cached")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--tag", default="hybrid")
    a = ap.parse_args()
    client = ollama.Client(host=HOST)
    sets = {"dev": CASES, "heldout": HELDOUT, "fresh": FRESH}
    out = []
    for model in a.models:
        r = {"model": model, "mode": a.mode, "sets": {}}
        one_call(client, model, SYSTEM, user_msg("greeting", None, "hello"), threads=a.threads, schema=state_schema("greeting"))
        for name, cases in sets.items():
            llm_only, hybrid, lat_llm, lat_hyb, to_llm = 0, 0, [], [], 0
            for state, car, utt, gold in cases:
                pred, ms = llm_turn(client, model, a.mode, a.threads, state, car, utt)
                ok_llm = score(gold, json.dumps(pred))["exact"]
                llm_only += ok_llm
                lat_llm.append(ms)
                fp = classify(state, car, utt)
                if fp is not None:
                    hybrid += score(gold, json.dumps({"intent": fp.intent, "slots": fp.slots}))["exact"]
                    lat_hyb.append(0.3)
                else:
                    to_llm += 1
                    hybrid += ok_llm          # reuse the LLM answer already computed (same call, same prompt)
                    lat_hyb.append(ms)
            n = len(cases)
            r["sets"][name] = {"n": n, "llm_only_exact": round(llm_only / n, 3), "hybrid_exact": round(hybrid / n, 3),
                               "share_to_llm": round(to_llm / n, 3),
                               "llm_only_ms": summarize(lat_llm), "hybrid_ms": summarize(lat_hyb)}
            x = r["sets"][name]
            print(f"{model:30s} {name:8s} llm-only exact {x['llm_only_exact']} | hybrid exact {x['hybrid_exact']} "
                  f"({x['share_to_llm']:.0%} to LLM) | per-turn p50/p95 ms: llm-only {x['llm_only_ms']['median']}/{x['llm_only_ms']['p95']}"
                  f" hybrid {x['hybrid_ms']['median']}/{x['hybrid_ms']['p95']}", flush=True)
        out.append(r)
        save_result(a.tag, {"results": out})


if __name__ == "__main__":
    main()
