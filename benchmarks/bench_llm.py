"""LLM benchmark (Ollama): speed with/without prefix cache, thread sweep, NLU accuracy.

Usage: uv run python benchmarks/bench_llm.py --models qwen3:0.6b qwen3:1.7b --accuracy
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import ollama

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import now, save_result, summarize  # noqa: E402
from nlu_cases import CASES, FRESH, HELDOUT, SCHEMA, SYSTEM, state_schema, state_system  # noqa: E402

HOST = "http://127.0.0.1:11434"
LICENSE_NOTE = {
    "qwen3": "Apache-2.0", "ministral-3": "Apache-2.0 (Ollama page)",
    "qwen3.5": "UNVERIFIED", "llama3.2": "Llama license: benchmark only",
    "gemma3": "Gemma terms: benchmark only",
}


@dataclass
class Call:
    ttft_ms: float
    total_ms: float
    prompt_eval_tokens: int
    prompt_eval_ms: float
    gen_tokens: int
    gen_tok_s: float
    text: str


def user_msg(state: str, car: str | None, utt: str) -> str:
    return f"STATE: {state}\nCAR_IN_FOCUS: {car or 'none'}\nCUSTOMER: {utt}"


def one_call(client: ollama.Client, model: str, system: str, user: str, *, threads: int | None,
             num_ctx: int = 2048, max_tokens: int = 96, think: bool = False, schema: dict | None = None) -> Call:
    opts = {"temperature": 0, "num_ctx": num_ctx, "num_predict": max_tokens}
    if threads:
        opts["num_thread"] = threads
    t0 = now()
    ttft = None
    text = []
    final = None
    kwargs = dict(model=model, stream=True, format=schema or SCHEMA, options=opts, keep_alive="30m",
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    # think=False is Ollama's switch for Qwen3-style models. Models that do not support
    # thinking may reject it, so retry without it only on that specific error.
    try:
        stream = client.chat(think=think, **kwargs)
        chunks = iter(stream)
        first = next(chunks)
    except ollama.ResponseError as e:
        if "think" not in str(e).lower():
            raise
        stream = client.chat(**kwargs)
        chunks = iter(stream)
        first = next(chunks)
    for ch in [first, *chunks]:
        piece = ch.message.content or ""
        if piece and ttft is None:
            ttft = (now() - t0) * 1000
        text.append(piece)
        if ch.done:
            final = ch
    total = (now() - t0) * 1000
    pe_n = final.prompt_eval_count or 0
    pe_ms = (final.prompt_eval_duration or 0) / 1e6
    g_n = final.eval_count or 0
    g_ms = (final.eval_duration or 1) / 1e6
    # With a JSON schema, Ollama delivers the output in one burst, so client-side TTFT == total.
    # Use the server's own timers instead: load + prompt evaluation + one generated token.
    # (prompt_eval_count does not shrink on cache hits, but prompt_eval_duration does.)
    per_tok = g_ms / g_n if g_n else 0.0
    ttft_est = (final.load_duration or 0) / 1e6 + pe_ms + per_tok
    return Call(ttft_est, total, pe_n, pe_ms, g_n, g_n / (g_ms / 1000) if g_ms else 0, "".join(text))


TOPIC_RE = {"warranty": r"warrant", "financing": r"financ|loan|bank|instal", "inspection": r"inspect",
            "trade_in": r"trade|exchange|old car", "documents": r"document|paper|bring|need"}


def ground_slots(utt: str, slots: dict) -> dict:
    """Deterministic filter: keep only slots whose value is actually supported by the utterance.

    Small models invent slot values ("offer_amount": 0, a made-up topic). Numbers must appear in
    the utterance (digits or words), strings must be substrings, topics need a keyword.
    """
    from stt_eval import canon_numbers, normalise  # noqa: PLC0415

    u = " " + normalise(utt) + " "
    nums = canon_numbers(utt)
    kept = {}
    for k, v in slots.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            if v != 0 and str(v) in nums:
                kept[k] = v
        elif k == "phone":
            d = "".join(c for c in str(v) if c.isdigit())
            if d and d in nums:
                kept[k] = v
        elif k == "topic":
            import re  # noqa: PLC0415

            if re.search(TOPIC_RE.get(str(v), r"$^"), utt.lower()):
                kept[k] = v
        else:
            nv = normalise(str(v))
            if nv and f" {nv} " in u:
                kept[k] = v
    return kept


def score(gold: dict, text: str, utt: str = "", grounded: bool = False) -> dict:
    try:
        pred = json.loads(text)
    except json.JSONDecodeError:
        return {"valid": False, "intent": False, "tp": 0, "fp": 0, "fn": len(gold["slots"]), "exact": False}
    pslots = pred.get("slots") or {}
    if grounded:
        pslots = ground_slots(utt, pslots)

    def norm(k, v):
        if k == "phone":
            return "".join(c for c in str(v) if c.isdigit())
        return v if isinstance(v, int) else str(v).strip().lower()

    def match(k, g, p):
        if k in ("date", "time", "name", "model", "make"):
            return g in p or p in g  # lenient: deterministic code normalises these later
        return g == p

    tp = fp = 0
    matched = set()
    for k, v in pslots.items():
        if k in gold["slots"] and match(k, norm(k, gold["slots"][k]), norm(k, v)):
            tp += 1
            matched.add(k)
        else:
            fp += 1
    fn = len([k for k in gold["slots"] if k not in matched])
    intent_ok = pred.get("intent") == gold["intent"]
    return {"valid": True, "intent": intent_ok, "tp": tp, "fp": fp, "fn": fn,
            "exact": intent_ok and fp == 0 and fn == 0}


def run_accuracy(client, model, threads, sp, cases) -> dict:
    """Run every case once, in order (states change between turns, as in a real call)."""
    rows, lat, pe = [], [], []
    for state, car, utt, gold in cases:
        s_, sc_ = sp(state)
        c = one_call(client, model, s_, user_msg(state, car, utt), threads=threads, schema=sc_)
        lat.append(c.total_ms)
        pe.append(c.prompt_eval_ms)
        g = score(gold, c.text, utt, grounded=True)
        rows.append({"state": state, "utt": utt, "gold": gold, "pred": c.text, "ms": round(c.total_ms),
                     "prefill_ms": round(c.prompt_eval_ms), **score(gold, c.text),
                     "g_exact": g["exact"], "g_tp": g["tp"], "g_fp": g["fp"], "g_fn": g["fn"]})
    n = len(rows)

    def f1(pre: str) -> float:
        tp = sum(r[pre + "tp"] for r in rows); fp = sum(r[pre + "fp"] for r in rows)
        fn = sum(r[pre + "fn"] for r in rows)
        p = tp / (tp + fp) if tp + fp else 1.0
        r_ = tp / (tp + fn) if tp + fn else 1.0
        return round(2 * p * r_ / (p + r_), 3) if p + r_ else 0.0

    return {
        "n": n,
        "json_valid": round(sum(r["valid"] for r in rows) / n, 3),
        "intent_acc": round(sum(r["intent"] for r in rows) / n, 3),
        "slot_f1": f1(""),
        "exact_match": round(sum(r["exact"] for r in rows) / n, 3),
        "slot_f1_grounded": f1("g_"),
        "exact_grounded": round(sum(r["g_exact"] for r in rows) / n, 3),
        "turn_ms": summarize(lat),          # whole NLU call per turn, state switches included
        "prefill_ms": summarize(pe),        # prompt evaluation per turn (cache hit or miss)
        "rows": rows,                       # every utterance, so results can be re-scored offline
    }


def bench_model(client, model: str, threads: int | None, repeat: int, accuracy: bool, mode: str = "full",
                case_sets: dict | None = None, speed: bool = True) -> dict:
    """mode 'full': one big prompt+schema. mode 'state': a short prompt+schema per conversation state."""
    out: dict = {"model": model, "threads": threads, "mode": mode}

    def sp(state: str) -> tuple[str, dict]:
        if mode == "full":      # one static prompt, one big schema
            return SYSTEM, SCHEMA
        if mode == "cached":    # one static prompt (always a cache hit), state enforced by the grammar only
            return SYSTEM, state_schema(state)
        return state_system(state), state_schema(state)   # 'state': short prompt per state
    base = model.split(":")[0]
    out["license"] = LICENSE_NOTE.get(base, "?")
    try:
        # Load + warm. First call includes model load, so keep it out of the stats.
        t0 = now()
        s0, sc0 = sp("negotiating")
        one_call(client, model, s0, user_msg("negotiating", "S1042", "hello"), threads=threads, schema=sc0)
        out["load_plus_first_call_ms"] = round((now() - t0) * 1000)

        # Uncached prefix: a unique first line breaks prefix reuse every time.
        cold = [] if not speed else [one_call(client, model, f"[{i}-{now()}]\n" + s0,
                         user_msg("negotiating", "S1042", "can you do forty eight thousand"), threads=threads, schema=sc0)
                for i in range(repeat)]
        # Cached prefix: identical system prompt, different customer text, same state.
        neg = [c for c in CASES if c[0] == "negotiating"]
        warm_inputs = (neg * 3)[:repeat + 1]
        warm = [] if not speed else [one_call(client, model, s0, user_msg(s, car, u), threads=threads, schema=sc0)
                                     for s, car, u, _ in warm_inputs]
        if speed:
            out["cold_prefix"] = {
                "ttft_ms": summarize([c.ttft_ms for c in cold]),
                "prompt_tokens": cold[-1].prompt_eval_tokens,
                "prompt_eval_ms": summarize([c.prompt_eval_ms for c in cold]),
                "total_ms": summarize([c.total_ms for c in cold]),
            }
            out["warm_prefix"] = {
                "ttft_ms": summarize([c.ttft_ms for c in warm[1:]]),
                "prompt_eval_ms": summarize([c.prompt_eval_ms for c in warm[1:]]),
                "gen_tok_s": summarize([c.gen_tok_s for c in warm[1:]]),
                "gen_tokens": summarize([float(c.gen_tokens) for c in warm[1:]]),
                "total_ms": summarize([c.total_ms for c in warm[1:]]),
            }
        if accuracy:
            out["accuracy"] = {}
            for set_name, cases in case_sets.items():
                out["accuracy"][set_name] = run_accuracy(client, model, threads, sp, cases)
    except Exception as e:  # keep going; record why a model could not be benchmarked
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--threads", type=int, nargs="*", default=[None], help="num_thread values to sweep")
    ap.add_argument("--repeat", type=int, default=6)
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--tag", default="llm")
    ap.add_argument("--mode", choices=["full", "state", "cached"], default="full")
    ap.add_argument("--cases", choices=["dev", "heldout", "fresh", "both", "all"], default="all")
    ap.add_argument("--no-speed", action="store_true", help="accuracy only (skip cold/warm timing)")
    a = ap.parse_args()
    case_sets = {}
    if a.cases in ("dev", "both", "all"):
        case_sets["dev"] = CASES
    if a.cases in ("heldout", "both", "all"):
        case_sets["heldout"] = HELDOUT
    if a.cases in ("fresh", "all"):
        case_sets["fresh"] = FRESH
    client = ollama.Client(host=HOST)
    results = []
    for m in a.models:
        for t in a.threads:
            print(f"== {m} threads={t}", flush=True)
            r = bench_model(client, m, t, a.repeat, a.accuracy, a.mode, case_sets, speed=not a.no_speed)
            results.append(r)
            if "error" in r:
                print("   ERROR", r["error"])
            else:
                if "warm_prefix" in r:
                    w, c = r["warm_prefix"], r["cold_prefix"]
                    print(f"   cold ttft {c['ttft_ms']['median']}ms ({c['prompt_tokens']} prompt tok) | "
                          f"warm ttft {w['ttft_ms']['median']}ms (prompt eval {w['prompt_eval_ms']['median']}ms) | "
                          f"{w['gen_tok_s']['median']} tok/s | warm total {w['total_ms']['median']}ms "
                          f"({w['gen_tokens']['median']} tok)", flush=True)
                for name, ac in r.get("accuracy", {}).items():
                    print(f"   [{name}] intent {ac['intent_acc']} exact {ac['exact_match']} | grounded exact "
                          f"{ac['exact_grounded']} slotF1 {ac['slot_f1_grounded']} | turn {ac['turn_ms']['median']}ms "
                          f"p95 {ac['turn_ms']['p95']}ms", flush=True)
            save_result(a.tag, {"results": results})
    print("saved", a.tag)


if __name__ == "__main__":
    main()
