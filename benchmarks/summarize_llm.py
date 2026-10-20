"""Turn the raw LLM suite JSON into the tables used in the Phase 0b report."""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

R = Path(__file__).parent / "results"
tag = sys.argv[1] if len(sys.argv) > 1 else "ac"


def load(name):
    p = R / f"{tag}_{name}.json"
    return {r["model"]: r for r in json.loads(p.read_text(encoding="utf-8"))["results"]} if p.exists() else {}


def med(*xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


p1, p2, cached = load("speed_state_p1"), load("speed_state_p2"), load("speed_cached")
accs, accc = load("acc_state"), load("acc_cached")
models = list(p1)

print("## Speed, per-state prompt (median of 2 passes; miss = prompt prefix not cached, hit = cached)")
print("| model | prompt tok | prefill miss ms | prefill hit ms | ttft miss | ttft hit | decode tok/s | gen tok | total hit ms | total miss ms |")
print("|---|---|---|---|---|---|---|---|---|---|")
speed = {}
for m in models:
    rs = [x[m] for x in (p1, p2) if m in x and "error" not in x[m]]
    if not rs:
        print(f"| {m} | ERROR: {p1[m].get('error')} |"); continue
    g = lambda *path: med(*[eval("r" + "".join(f"['{k}']" for k in path)) for r in rs])
    d = dict(tok=rs[0]["cold_prefix"]["prompt_tokens"],
             pm=g("cold_prefix", "prompt_eval_ms", "median"), ph=g("warm_prefix", "prompt_eval_ms", "median"),
             tm=g("cold_prefix", "ttft_ms", "median"), th=g("warm_prefix", "ttft_ms", "median"),
             tps=g("warm_prefix", "gen_tok_s", "median"), gen=g("warm_prefix", "gen_tokens", "median"),
             tot_h=g("warm_prefix", "total_ms", "median"), tot_m=g("cold_prefix", "total_ms", "median"))
    speed[m] = d
    print(f"| {m} | {d['tok']} | {d['pm']:.0f} | {d['ph']:.0f} | {d['tm']:.0f} | {d['th']:.0f} | {d['tps']:.1f} | {d['gen']:.0f} | {d['tot_h']:.0f} | {d['tot_m']:.0f} |")

print("\n## Speed, static prompt + grammar-only state ('cached' mode)")
print("| model | prompt tok | prefill miss ms | prefill hit ms | decode tok/s | total hit ms | total miss ms |")
print("|---|---|---|---|---|---|---|")
for m, r in cached.items():
    if "error" in r:
        print(f"| {m} | ERROR {r['error'][:60]} |"); continue
    c, w = r["cold_prefix"], r["warm_prefix"]
    print(f"| {m} | {c['prompt_tokens']} | {c['prompt_eval_ms']['median']:.0f} | {w['prompt_eval_ms']['median']:.0f} | "
          f"{w['gen_tok_s']['median']:.1f} | {w['total_ms']['median']:.0f} | {c['total_ms']['median']:.0f} |")

for title, accd in (("per-state prompt", accs), ("static prompt + grammar-only state ('cached')", accc)):
    print(f"\n## Accuracy, {title}  (intent / exact / exact after grounding filter)")
    print("| model | dev (31) | heldout (40) | fresh (29) | all 100 exact+grounded | turn p50 ms | turn p95 ms | prefill p50 ms |")
    print("|---|---|---|---|---|---|---|---|")
    for m, r in accd.items():
        if "error" in r:
            print(f"| {m} | ERROR {r['error'][:70]} |"); continue
        a = r["accuracy"]
        cell = lambda s: f"{a[s]['intent_acc']:.2f} / {a[s]['exact_match']:.2f} / {a[s]['exact_grounded']:.2f}"
        rows = sum((a[s]["rows"] for s in a), [])
        ex = sum(x["g_exact"] for x in rows) / len(rows)
        lat = sorted(x["ms"] for x in rows); pf = sorted(x["prefill_ms"] for x in rows)
        q = lambda v, p: v[min(len(v) - 1, int(len(v) * p))]
        print(f"| {m} | {cell('dev')} | {cell('heldout')} | {cell('fresh')} | {ex:.2f} | {q(lat, .5)} | {q(lat, .95)} | {q(pf, .5)} |")
