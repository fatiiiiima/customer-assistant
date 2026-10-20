"""Run STT, LLM and TTS workloads alone and then all at once; report the slowdown.

Worst-case overlap (barge-in: caller talks while the agent speaks while the LLM runs).
Each worker is its own process with its own thread budget, like separate services.
Usage:
  uv run python benchmarks/bench_concurrent.py --stt sherpa:<dir> --llm qwen3:1.7b \
      --tts piper:<dir> --stt-threads 3 --tts-threads 3 --llm-threads 6 --seconds 25
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def stt_worker(spec, threads, seconds, barrier, q):
    from bench_stt import build, pick_clips
    from common import load_wav_16k

    clips, _ = pick_clips()
    audio = load_wav_16k(dict(clips).get("num01", clips[0][1]))
    eng = build(spec, threads)
    eng.transcribe(audio)
    barrier.wait()
    end, lats = time.perf_counter() + seconds, []
    while time.perf_counter() < end:
        lats.append(eng.transcribe(audio)[1])
    q.put(("stt", lats))


def tts_worker(spec, threads, seconds, barrier, q):
    from bench_tts import MEDIUM, build

    eng = build(spec, threads)
    eng.run(MEDIUM)
    barrier.wait()
    end, lats = time.perf_counter() + seconds, []
    while time.perf_counter() < end:
        lats.append(eng.run(MEDIUM)[0])  # TTFA
    q.put(("tts", lats))


def llm_worker(model, threads, seconds, barrier, q):
    import ollama
    from bench_llm import one_call, user_msg
    from nlu_cases import CASES, state_schema, state_system

    client = ollama.Client(host="http://127.0.0.1:11434")
    neg = [c for c in CASES if c[0] == "negotiating"]
    sysm, sc = state_system("negotiating"), state_schema("negotiating")
    one_call(client, model, sysm, user_msg("negotiating", "S1042", "hello"), threads=threads, schema=sc)
    barrier.wait()
    end, lats, i = time.perf_counter() + seconds, [], 0
    while time.perf_counter() < end:
        s, car, u, _ = neg[i % len(neg)]
        i += 1
        lats.append(one_call(client, model, sysm, user_msg(s, car, u), threads=threads, schema=sc).total_ms)
    q.put(("llm", lats))


def run(workers: dict, seconds: int) -> dict:
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(len(workers) + 1), ctx.Queue()
    procs = [ctx.Process(target=fn, args=(*args, seconds, barrier, q)) for fn, args in workers.values()]
    for p in procs:
        p.start()
    barrier.wait()  # all models loaded and warm
    out = {}
    for _ in procs:
        name, lats = q.get(timeout=seconds + 120)
        out[name] = lats
    for p in procs:
        p.join()
    return out


def main() -> None:
    from common import save_result, summarize

    ap = argparse.ArgumentParser()
    ap.add_argument("--stt", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--tts", required=True)
    ap.add_argument("--stt-threads", type=int, default=3)
    ap.add_argument("--tts-threads", type=int, default=3)
    ap.add_argument("--llm-threads", type=int, default=6)
    ap.add_argument("--seconds", type=int, default=25)
    ap.add_argument("--stages", default="stt,llm,tts", help="comma list; e.g. stt,tts = barge-in overlap with LLM idle")
    ap.add_argument("--tag", default="concurrent")
    a = ap.parse_args()
    ws = {
        "stt": (stt_worker, (a.stt, a.stt_threads)),
        "llm": (llm_worker, (a.llm, a.llm_threads)),
        "tts": (tts_worker, (a.tts, a.tts_threads)),
    }
    ws = {k: v for k, v in ws.items() if k in a.stages.split(",")}
    solo = {k: summarize(run({k: v}, a.seconds)[k]) for k, v in ws.items()}
    together = {k: summarize(v) for k, v in run(ws, a.seconds).items()}
    print(f"{'stage':5s} {'solo p50':>9s} {'solo p95':>9s} {'all p50':>9s} {'all p95':>9s} {'x slower (p50)':>15s}")
    for k in ws:
        s, t = solo[k], together[k]
        print(f"{k:5s} {s['median']:9.0f} {s['p95']:9.0f} {t['median']:9.0f} {t['p95']:9.0f} {t['median'] / s['median']:15.2f}")
    save_result(a.tag, {"config": vars(a), "solo": solo, "together": together})


if __name__ == "__main__":
    main()
