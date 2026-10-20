"""Does OpenVINO GenAI reuse the KV cache across generate() calls? Compare a repeated system prompt (hit?)
with a prompt whose FIRST line is unique every call (guaranteed miss). Intent-only constrained call."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import MODELS, now, save_result, summarize  # noqa: E402
from nlu_cases import CASES, state_schema, state_system  # noqa: E402
import openvino_genai as ov  # noqa: E402

neg = [c for c in CASES if c[0] == "negotiating"]
out = []
for model, device in (("Qwen3-1.7B-int4-ov", "CPU"), ("Qwen3-1.7B-int4-ov", "GPU"), ("Qwen3-4B-int4-ov", "GPU"), ("Qwen3-4B-int4-ov", "CPU")):
    pipe = ov.LLMPipeline(str(MODELS / "ov" / model), device, CACHE_DIR=str(MODELS / "ov_cache"))
    tok = pipe.get_tokenizer()
    cfg = ov.GenerationConfig(); cfg.max_new_tokens, cfg.do_sample = 14, False
    sch = state_schema("negotiating")
    cfg.structured_output_config = ov.StructuredOutputConfig(json_schema=json.dumps({"type": "object", "properties": {"intent": sch["properties"]["intent"]}, "required": ["intent"]}))

    def run(i, nonce):
        s, car, u, _ = neg[i % len(neg)]
        sysm = (f"[{i}-{now()}]\n" if nonce else "") + state_system("negotiating")
        p = tok.apply_chat_template([{"role": "system", "content": sysm}, {"role": "user", "content": f"STATE: {s}\nCAR_IN_FOCUS: {car}\nCUSTOMER: {u}"}], add_generation_prompt=True, extra_context={"enable_thinking": False})
        t = now(); r = pipe.generate([p], cfg); wall = (now() - t) * 1000
        pm = r.perf_metrics
        return pm.get_ttft().mean, wall, pm.get_num_input_tokens()

    run(0, False)
    same = [run(i, False) for i in range(1, 8)]
    miss = [run(i, True) for i in range(1, 8)]
    row = {"model": model, "device": device,
           "repeated_prefix_ttft": summarize([x[0] for x in same]), "repeated_total": summarize([x[1] for x in same]),
           "unique_prefix_ttft": summarize([x[0] for x in miss]), "unique_total": summarize([x[1] for x in miss]),
           "prompt_tokens": same[0][2]}
    out.append(row)
    print(f"{model:20s} {device:4s} | repeated prefix: TTFT {row['repeated_prefix_ttft']['median']:6.0f} ms total {row['repeated_total']['median']:6.0f} | "
          f"UNIQUE prefix (miss): TTFT {row['unique_prefix_ttft']['median']:6.0f} ms total {row['unique_total']['median']:6.0f} | {row['prompt_tokens']} tok", flush=True)
    del pipe
save_result("diag_ov_cache", {"results": out})
