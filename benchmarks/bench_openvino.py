"""OpenVINO benchmark on CPU / iGPU (GPU) / NPU for the LLM and Whisper.

Does not assume acceleration helps: every device is timed on the same prompt, and failures
(unsupported model on NPU, compile errors) are recorded rather than hidden.
LLM: Qwen3 int4 OpenVINO builds from the OpenVINO org on Hugging Face (Apache-2.0 base model).
  TTFT here = full prompt prefill (no prefix cache), the worst case; decode = tok/s.
Whisper: OpenVINO/whisper-{tiny,base}-int8-ov (MULTILINGUAL builds; no .en variants exist).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import MODELS, RESULTS, load_wav_16k, now, save_result, summarize  # noqa: E402
from nlu_cases import CASES, state_schema, state_system  # noqa: E402

import openvino_genai as ov_genai  # noqa: E402

CACHE = str(MODELS / "ov_cache")


def bench_llm(model_dir: str, device: str, repeat: int, structured: bool, intent_only: bool = False) -> dict:
    out: dict = {"model": model_dir, "device": device, "structured": structured, "intent_only": intent_only}
    try:
        t0 = now()
        pipe = ov_genai.LLMPipeline(str(MODELS / "ov" / model_dir), device, CACHE_DIR=CACHE)
        out["load_s"] = round(now() - t0, 1)
        tok = pipe.get_tokenizer()
        neg = [c for c in CASES if c[0] == "negotiating"]
        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = 14 if intent_only else 40
        cfg.do_sample = False
        if structured:
            sch = state_schema("negotiating")
            if intent_only:  # the intent field comes first, so this yields the same intent as the full JSON
                sch = {"type": "object", "properties": {"intent": sch["properties"]["intent"]}, "required": ["intent"]}
            cfg.structured_output_config = ov_genai.StructuredOutputConfig(json_schema=json.dumps(sch))
        ttft, tps, total, ntok, nin = [], [], [], [], []
        for i in range(repeat + 1):
            s, car, u, _ = neg[i % len(neg)]
            history = [{"role": "system", "content": state_system("negotiating")},
                       {"role": "user", "content": f"STATE: {s}\nCAR_IN_FOCUS: {car}\nCUSTOMER: {u}"}]
            prompt = tok.apply_chat_template(history, add_generation_prompt=True,
                                             extra_context={"enable_thinking": False})
            t = now()
            res = pipe.generate([prompt], cfg)  # list input -> DecodedResults (a bare str returns only text, no perf_metrics)
            wall = (now() - t) * 1000
            if i == 0:
                continue  # warm-up (kernel compile / caches)
            pm = res.perf_metrics
            ttft.append(pm.get_ttft().mean)
            tps.append(pm.get_throughput().mean)
            total.append(wall)
            ntok.append(float(pm.get_num_generated_tokens()))
            nin.append(float(pm.get_num_input_tokens()))
        out.update(ttft_ms=summarize(ttft), decode_tok_s=summarize(tps), total_ms=summarize(total),
                   gen_tokens=summarize(ntok), prompt_tokens=int(nin[0]), sample=str(res.texts[0])[:160])
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return out


def bench_whisper(model_dir: str, device: str, repeat: int) -> dict:
    out: dict = {"model": model_dir, "device": device}
    try:
        from bench_stt import pick_clips  # noqa: PLC0415  (imports only helpers)

        clips, kind = pick_clips()
        audio = load_wav_16k(dict(clips).get("num01", clips[0][1]))
        t0 = now()
        pipe = ov_genai.WhisperPipeline(str(MODELS / "ov" / model_dir), device, CACHE_DIR=CACHE)
        out["load_s"] = round(now() - t0, 1)
        cfg = pipe.get_generation_config()
        cfg.language = "<|en|>"
        cfg.task = "transcribe"
        lats = []
        for i in range(repeat + 1):
            t = now()
            res = pipe.generate(audio.tolist(), cfg)
            if i:
                lats.append((now() - t) * 1000)
        out.update(latency_ms=summarize(lats), clip_s=round(len(audio) / 16000, 2), text=str(res)[:120])
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", nargs="*", default=["Qwen3-0.6B-int4-ov", "Qwen3-1.7B-int4-ov"])
    ap.add_argument("--whisper", nargs="*", default=["whisper-tiny-int8-ov", "whisper-base-int8-ov"])
    ap.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"])
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--intent-only", action="store_true", help="LLM outputs only the intent (slots come from code)")
    ap.add_argument("--structured", action="store_true", help="also run LLM with JSON-schema constrained decoding")
    ap.add_argument("--tag", default="openvino")
    a = ap.parse_args()
    res = {"llm": [], "whisper": []}
    for m in a.llm:
        for d in a.devices:
            for st in ([False, True] if a.structured else [False]):
                r = bench_llm(m, d, a.repeat, st or a.intent_only, a.intent_only)
                res["llm"].append(r)
                print(f"LLM {m} {d} structured={st}:",
                      r.get("error") or f"load {r['load_s']}s | TTFT {r['ttft_ms']['median']}ms "
                      f"({r['prompt_tokens']} prompt tok) | {r['decode_tok_s']['median']:.1f} tok/s | total {r['total_ms']['median']}ms",
                      flush=True)
                save_result(a.tag, res)
    for m in a.whisper:
        for d in a.devices:
            r = bench_whisper(m, d, a.repeat)
            res["whisper"].append(r)
            print(f"WHISPER {m} {d}:", r.get("error") or f"load {r['load_s']}s | {r['latency_ms']['median']}ms for {r['clip_s']}s clip | {r['text']!r}",
                  flush=True)
            save_result(a.tag, res)


if __name__ == "__main__":
    main()

