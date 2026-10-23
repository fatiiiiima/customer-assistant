r"""Many callers asking the model at once: today's single pipeline (one request at a time) vs OpenVINO continuous batching with prefix caching.

  .venv\Scripts\python.exe scripts\bench_llm_concurrency.py                 # levels 1 4 8, real prompts from logs/prompts.json

For each level N, N requests start together (each a real agent prompt with a different caller line). Reported: time to the first token and to
the last token (median / worst). Nothing else may use the iGPU while this runs.
"""
from __future__ import annotations

import json
import statistics as st
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "ov" / "Qwen3-4B-Instruct-2507-int4-ov"
LINES = ["do you have any Toyota", "how much is the second one", "can I book a test drive tomorrow", "what colour is it", "is the price negotiable",
         "how many kilometres has it done", "show me something cheaper", "I want an SUV under one hundred thousand"]


def prompts(tok) -> list[str]:
    ps = [p for p in json.loads((ROOT / "logs" / "prompts.json").read_text(encoding="utf-8")) if p.get("kind") != "decide"] or \
        json.loads((ROOT / "logs" / "prompts.json").read_text(encoding="utf-8"))
    base = ps[0]
    return [tok.apply_chat_template([{"role": "system", "content": base["system"]}, {"role": "user", "content": (base.get("user") or "") + "\nCALLER: " + line}],
                                    add_generation_prompt=True) for line in LINES]


def cfg(g, n_tokens: int):
    c = g.GenerationConfig()
    c.do_sample, c.max_new_tokens = False, n_tokens
    return c


def run_serial(g, pipe, ps: list[str], n: int, n_tokens: int) -> list[tuple[float, float]]:
    """Today: one pipeline, one request at a time (the others wait in line)."""
    t0, out, lock = time.perf_counter(), [], threading.Lock()

    def one(p):
        first = []

        def streamer(_):
            if not first:
                first.append(time.perf_counter() - t0)
            return g.StreamingStatus.RUNNING
        with lock:
            pipe.generate(p, cfg(g, n_tokens), streamer)
        out.append((first[0] if first else float("nan"), time.perf_counter() - t0))
    ts = [threading.Thread(target=one, args=(ps[i % len(ps)],)) for i in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return out


def run_batched(g, cb, ps: list[str], n: int, n_tokens: int, rid: list[int]) -> list[tuple[float, float]]:
    """Continuous batching: every request is added at once; one loop steps them all together."""
    t0 = time.perf_counter()
    handles = []
    for i in range(n):
        rid[0] += 1
        handles.append(cb.add_request(rid[0], ps[i % len(ps)], cfg(g, n_tokens)))
    first, last = [None] * n, [None] * n
    while cb.has_non_finished_requests():
        cb.step()
        now = time.perf_counter() - t0
        for i, h in enumerate(handles):
            if first[i] is None and h.can_read():
                first[i] = now
            if last[i] is None and h.get_status() != g.GenerationStatus.RUNNING:
                last[i] = now
    for h in handles:
        h.read_all() if hasattr(h, "read_all") else None
    now = time.perf_counter() - t0
    return [(f or now, l or now) for f, l in zip(first, last)]


def show(label: str, n: int, res: list[tuple[float, float]]) -> dict:
    f, l = [r[0] for r in res], [r[1] for r in res]
    row = {"setup": label, "requests": n, "first_median": round(st.median(f), 2), "first_worst": round(max(f), 2), "done_median": round(st.median(l), 2), "done_worst": round(max(l), 2)}
    print(f"{label:22s} {n} at once: first token median {row['first_median']} s, worst {row['first_worst']} s; finished median {row['done_median']} s, worst {row['done_worst']} s", flush=True)
    return row


def main() -> int:
    import openvino_genai as g

    levels = [int(x) for x in sys.argv[1:]] or [1, 4, 8]
    n_tokens = 40
    rows = []
    pipe = g.LLMPipeline(str(MODEL), "GPU", CACHE_DIR=str(ROOT / "models" / "ov_cache"))
    ps = prompts(pipe.get_tokenizer())
    pipe.generate(ps[0], cfg(g, 4))
    for n in levels:
        rows.append(show("one at a time (today)", n, run_serial(g, pipe, ps, n, n_tokens)))
    del pipe
    sc = g.SchedulerConfig()
    sc.enable_prefix_caching = True
    sc.cache_size = 2                                   # GB of KV cache on the iGPU
    sc.max_num_batched_tokens = 4096
    cb = g.ContinuousBatchingPipeline(str(MODEL), sc, "GPU", {"CACHE_DIR": str(ROOT / "models" / "ov_cache")})
    rid = [0]
    run_batched(g, cb, ps, 1, 4, rid)
    for n in levels:
        rows.append(show("batched + prefix cache", n, run_batched(g, cb, ps, n, n_tokens, rid)))
    (ROOT / "logs" / "llm_concurrency.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
