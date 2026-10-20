"""T0: compare NLU backends/checkpoints on the same labelled utterances.

  uv run python -m dealer_core.eval.compare --sets dev heldout fresh \\
      --spec ollama_orig=ollama:qwen3:4b-q4_K_M::state --spec ov_orig=openvino:Qwen3-4B-int4-ov:GPU:state \\
      --pair ov_orig:ollama_orig

Spec format  label=runtime:model:device:prompt_mode   (device empty for ollama; the model may contain ':').
Every backend answers each utterance in FULL-JSON mode (intent + slots), so the comparison is apples to apples, and
also in the production INTENT-ONLY mode. Predictions are cached in <out>/<label>.json, so re-scoring is free and a
growing labelled set only costs the new rows.

A pair "cand:base" is compared on exact+ground with a paired bootstrap interval and an exact McNemar test.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ..config import BackendCfg, LlmCfg, load_config
from ..nlu.backend import BackendError
from ..nlu.chain import make_backend
from .datasets import Example, load_sets
from .scoring import score
from .stats import gate_verdict, paired_compare

ROOT = Path(__file__).resolve().parents[3]


def parse_spec(spec: str) -> tuple[str, BackendCfg]:
    label, _, rest = spec.partition("=")
    runtime, _, rest = rest.partition(":")
    model_dev_mode = rest.rsplit(":", 2)          # model may itself contain ':' (ollama tags)
    model, device, mode = model_dev_mode[0], model_dev_mode[1], model_dev_mode[2]
    return label, BackendCfg(runtime=runtime, model=model, device=device or None, prompt_mode=mode or "auto", timeout_s=30.0)


def run_backend(label: str, bcfg: BackendCfg, examples: list[Example], out_dir: Path, rerun: bool, llm: LlmCfg) -> dict[str, dict]:
    cache_path = out_dir / f"{label}.json"
    cache: dict[str, dict] = {} if rerun or not cache_path.exists() else json.loads(cache_path.read_text(encoding="utf-8"))
    todo = [e for e in examples if e.id not in cache]
    if todo:
        backend = make_backend(label, bcfg, LlmCfg(backend=label, text_backend=label, fallbacks=[], backends={label: bcfg},
                                                    think=llm.think, temperature=llm.temperature), ROOT / "models")
        h = backend.health()
        if not h.ok:
            raise BackendError(f"{label}: {h.detail}")
        print(f"[{label}] {h.detail}: running {len(todo)} utterances", flush=True)
        for n, e in enumerate(todo, 1):
            t0 = time.perf_counter()
            try:
                js = backend.complete_json(e.state, e.car, e.utterance)
            except BackendError as err:
                js = {"error": str(err)}
            t1 = time.perf_counter()
            try:
                io = backend.classify(e.state, e.car, e.utterance).intent.value
            except BackendError:
                io = None
            cache[e.id] = {"json": js, "intent_only": io, "json_ms": (t1 - t0) * 1000, "intent_ms": (time.perf_counter() - t1) * 1000}
            if n % 25 == 0:
                print(f"   {n}/{len(todo)}", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        del backend
    return cache


def summarise(label: str, examples: list[Example], preds: dict[str, dict]) -> dict:
    out: dict = {"label": label, "sets": {}}
    by_set: dict[str, list[Example]] = {}
    for e in examples:
        by_set.setdefault(e.set, []).append(e)
    by_set["ALL"] = examples
    for name, exs in by_set.items():
        sc = [score(e, preds[e.id]["json"], ground=True) for e in exs]
        raw = [score(e, preds[e.id]["json"], ground=False) for e in exs]
        io = [preds[e.id]["intent_only"] == e.gold_intent.value for e in exs]
        lenient = [(preds[e.id]["json"].get("intent") in {e.gold_intent.value, *e.also_ok}) for e in exs]
        agree = [preds[e.id]["intent_only"] == preds[e.id]["json"].get("intent") for e in exs]
        n = len(exs)
        out["sets"][name] = {
            "n": n, "n_ambiguous": sum(e.ambiguous for e in exs), "n_state_guess": sum(e.state_guess for e in exs), "intent_lenient": sum(lenient) / n,
            "intent": sum(s.intent_ok for s in sc) / n, "exact": sum(s.exact for s in raw) / n,
            "exact_ground": sum(s.exact for s in sc) / n, "intent_only_acc": sum(io) / n, "intent_only_agrees": sum(agree) / n,
            "json_ms_p50": statistics.median(preds[e.id]["json_ms"] for e in exs),
            "intent_ms_p50": statistics.median(preds[e.id]["intent_ms"] for e in exs)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["dev", "heldout", "fresh"])
    ap.add_argument("--csv", nargs="*", type=Path, default=[], help="labelled CSV sets (your utterances)")
    ap.add_argument("--spec", action="append", required=True)
    ap.add_argument("--pair", action="append", default=[], help="candidate:baseline (labels)")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results" / "t0")
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()

    sets = load_sets(ROOT / "data" / "nlu_sets", a.sets, a.csv)
    examples = [e for exs in sets.values() for e in exs]
    llm = load_config(environ={}).llm
    results, flags = {}, {}
    for spec in a.spec:
        label, bcfg = parse_spec(spec)
        preds = run_backend(label, bcfg, examples, a.out, a.rerun, llm)
        results[label] = summarise(label, examples, preds)
        flags[label] = [score(e, preds[e.id]["json"], ground=True).exact for e in examples]

    print("\n== exact+ground (intent+slots, grounding filter on) / intent accuracy / intent-only accuracy ==")
    names = list(next(iter(results.values()))["sets"])
    print(f"{'config':18s} " + " ".join(f"{n:>22s}" for n in names))
    for label, r in results.items():
        cells = [f"{r['sets'][n]['exact_ground']:.2f}/{r['sets'][n]['intent']:.2f}/{r['sets'][n]['intent_only_acc']:.2f}" for n in names]
        print(f"{label:18s} " + " ".join(f"{c:>22s}" for c in cells))
    print("\n== latency p50 per utterance (full JSON / intent-only), ms: for reference, not the latency benchmark ==")
    for label, r in results.items():
        a_ = r["sets"]["ALL"]
        print(f"{label:18s} {a_['json_ms_p50']:7.0f} / {a_['intent_ms_p50']:6.0f}   intent-only agrees with full-JSON intent: {a_['intent_only_agrees']:.0%}")

    pairs = {}
    for pair in a.pair:
        cand, base = pair.split(":")
        r = paired_compare(flags[cand], flags[base])
        verdict = gate_verdict(r)
        pairs[pair] = {**r.__dict__, "verdict": verdict}
        print(f"\n{cand} vs {base} on {r.n} utterances: exact+ground {r.diff_pts:+.1f} pts, 95% CI [{r.ci_low:+.1f}, {r.ci_high:+.1f}]; "
              f"McNemar {cand}-only {r.cand_only} / {base}-only {r.base_only}, p={r.p_mcnemar:.2f}\n   -> {verdict}")
    (a.out / "summary.json").write_text(json.dumps({"results": results, "pairs": pairs}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
