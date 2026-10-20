"""Is missing prefix-cache reuse caused by the MODEL/runtime or by MY SETUP?

For each model, send controlled call sequences and compare prompt-evaluation time (the part
a cache hit removes). Each variant changes exactly one thing relative to variant B (my
benchmark's pattern: same system prompt, different user text, identical options each call).

  A  identical request repeated (system, user, options all identical)  -> best case for a cache
  B  same system prompt, different user text                           -> my benchmark pattern
  C  B without the JSON-schema `format`                                -> is the grammar to blame?
  D  B without the think= parameter                                    -> does think= change the prompt?
  E  B with a much longer shared prefix (~900 tokens)                  -> does reuse scale with prefix?
  F  B with different num_predict per call                             -> do changing options reset the slot?

Reading the result: if A also shows no speed-up, the model/runtime cannot reuse the cache at all
(setup is not the cause). If A speeds up but B does not, my prompt/options are changing the prefix.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).parent))
from common import save_result, summarize  # noqa: E402
from nlu_cases import CASES, state_schema, state_system  # noqa: E402

SYSTEM = state_system("negotiating")
SCHEMA = state_schema("negotiating")
USERS = [f"STATE: negotiating\nCAR_IN_FOCUS: S1042\nCUSTOMER: {c[2]}" for c in CASES if c[0] == "negotiating"]


def call(client, model, system, user, *, fmt=True, think=False, num_predict=48):
    kw = dict(model=model, stream=False, keep_alive="30m",
              options={"temperature": 0, "num_ctx": 4096, "num_predict": num_predict},
              messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    if fmt:
        kw["format"] = SCHEMA
    if think is not None:
        kw["think"] = think
    r = client.chat(**kw)
    return {"prefill_ms": (r.prompt_eval_duration or 0) / 1e6, "prompt_tokens": r.prompt_eval_count or 0,
            "gen_tokens": r.eval_count or 0}


def seq(client, model, n, make_args, **kw):
    runs = [call(client, model, *make_args(i), **kw) for i in range(n + 1)]
    first, rest = runs[0], runs[1:]
    return {"first_prefill_ms": round(first["prefill_ms"]), "later_prefill_ms": summarize([r["prefill_ms"] for r in rest]),
            "later_prompt_tokens": [r["prompt_tokens"] for r in rest][:4]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    client = ollama.Client(host="http://127.0.0.1:11434")
    long_sys = SYSTEM + "\n" + "\n".join(f"Example {i}: customer says '{u[-40:]}' -> intent make_offer." for i, u in enumerate(USERS * 12))
    results = []
    for m in a.models:
        info = client.show(m)
        r = {"model": m, "architecture": getattr(info.details, "family", None),
             "params": getattr(info.details, "parameter_size", None), "variants": {}}
        client.chat(model=m, messages=[{"role": "user", "content": "hi"}], think=False, options={"num_predict": 1, "num_ctx": 4096})
        v = r["variants"]
        v["A_identical_request"] = seq(client, m, a.n, lambda i: (SYSTEM, USERS[0]))
        v["B_same_system_diff_user"] = seq(client, m, a.n, lambda i: (SYSTEM, USERS[(i + 1) % len(USERS)]))
        v["C_no_json_schema"] = seq(client, m, a.n, lambda i: (SYSTEM, USERS[(i + 1) % len(USERS)]), fmt=False)
        v["D_no_think_param"] = seq(client, m, a.n, lambda i: (SYSTEM, USERS[(i + 1) % len(USERS)]), think=None)
        v["E_long_prefix"] = seq(client, m, a.n, lambda i: (long_sys, USERS[(i + 1) % len(USERS)]))
        runs = [call(client, m, SYSTEM, USERS[(i + 1) % len(USERS)], num_predict=40 + i * 8) for i in range(a.n + 1)]
        v["F_varying_num_predict"] = {"first_prefill_ms": round(runs[0]["prefill_ms"]),
                                      "later_prefill_ms": summarize([x["prefill_ms"] for x in runs[1:]])}
        results.append(r)
        print(f"== {m} ({r['architecture']}, {r['params']})")
        for k, x in v.items():
            print(f"   {k:26s} first {x['first_prefill_ms']:>5}ms  later median {x['later_prefill_ms']['median']}ms")
        save_result("diag_cache", {"results": results})


if __name__ == "__main__":
    main()
