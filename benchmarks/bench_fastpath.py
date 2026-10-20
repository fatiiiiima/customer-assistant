"""Evaluate the deterministic fast path: coverage, precision when it fires, and latency.

coverage  = share of utterances the rules answer (the rest go to the LLM)
precision = share of answered utterances that are exactly right (intent AND slots)
Reported separately for DEV (cases the rules were written while looking at) and HELDOUT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_llm import score  # noqa: E402
from common import now, save_result, summarize  # noqa: E402
from fastpath import classify  # noqa: E402
from nlu_cases import CASES, FRESH, HELDOUT  # noqa: E402


def evaluate(name: str, cases) -> dict:
    handled, correct, lat, wrong, deferred = 0, 0, [], [], []
    for state, car, utt, gold in cases:
        t = now()
        r = classify(state, car, utt)
        lat.append((now() - t) * 1000)
        if r is None:
            deferred.append({"utt": utt, "gold_intent": gold["intent"]})
            continue
        handled += 1
        s = score(gold, json.dumps({"intent": r.intent, "slots": r.slots}))
        if s["exact"]:
            correct += 1
        else:
            wrong.append({"utt": utt, "gold": gold, "got": {"intent": r.intent, "slots": r.slots}, "rule": r.rule})
    n = len(cases)
    out = {"n": n, "handled": handled, "coverage": round(handled / n, 3),
           "precision": round(correct / handled, 3) if handled else None,
           "latency_ms": summarize(lat), "wrong": wrong, "deferred": deferred}
    print(f"[{name}] coverage {handled}/{n} = {out['coverage']} | precision {correct}/{handled} | "
          f"median {out['latency_ms']['median']}ms p95 {out['latency_ms']['p95']}ms")
    for w in wrong:
        print(f"   WRONG  {w['utt']!r}\n          gold={w['gold']} got={w['got']} ({w['rule']})")
    for d in deferred:
        print(f"   defer  {d['utt']!r}  [{d['gold_intent']}]")
    return out


if __name__ == "__main__":
    res = {"dev": evaluate("dev", CASES), "heldout": evaluate("heldout", HELDOUT), "fresh": evaluate("fresh", FRESH)}
    save_result("fastpath", res)
