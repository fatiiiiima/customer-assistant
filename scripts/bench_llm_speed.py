r"""Decode speed of a local OpenVINO LLM on this machine, with and without speculative decoding (a small draft model proposes tokens, the big one checks them).

  .venv\Scripts\python.exe scripts\bench_llm_speed.py --model Qwen3-4B-Instruct-2507-int4-ov-nocal --draft Qwen3-0.6B-int4-ov --device GPU
  .venv\Scripts\python.exe scripts\bench_llm_speed.py --model Phi-4-mini-instruct-int4-ov

Prints time to first token, tokens per second, and the text, for a reply-sized prompt of the kind the voice agent sends (a few facts, ask for a short spoken answer).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import openvino_genai as ov

ROOT = Path(__file__).resolve().parents[1]
PROMPT_SYS = ("You are the voice of a warm sales assistant at AGMC, a used-car dealership, on the phone. Answer in one or two short spoken sentences, "
              "using only the FACTS. No emojis.")
PROMPT_USER = ("CUSTOMER: how much is the Camry and how many kilometres has it done?\nFACTS: 2020 Toyota Camry, brown, 52,000 kilometres, 74,500 dirhams, "
               "hybrid, automatic, full service history, at the AGMC showroom on Sheikh Zayed Road.\nSAY:")


def chat_prompt(tok, think_off: bool) -> str:
    msgs = [{"role": "system", "content": PROMPT_SYS}, {"role": "user", "content": PROMPT_USER}]
    try:
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, extra_context={"enable_thinking": False})
    except Exception:
        p = tok.apply_chat_template(msgs, add_generation_prompt=True)
    return p


def run(pipe, prompt: str, n_tokens: int, assistant: int | None, tok=None) -> tuple[float, float, str]:
    cfg = ov.GenerationConfig()
    cfg.max_new_tokens, cfg.do_sample = n_tokens, False
    if assistant:
        cfg.num_assistant_tokens = assistant
    first = []
    t0 = time.perf_counter()
    count = [0]

    def streamer(piece: str):
        if not first:
            first.append(time.perf_counter() - t0)
        count[0] += 1
        return ov.StreamingStatus.RUNNING

    out = str(pipe.generate(prompt, cfg, streamer))
    total = time.perf_counter() - t0
    n_out = len(tok.encode(out).input_ids.data[0]) if tok is not None else count[0]       # real token count (speculative decoding calls back in bursts)
    tps = (n_out - 1) / max(total - first[0], 1e-6) if first and n_out > 1 else 0.0
    return (first[0] if first else total), tps, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="folder under models/ov")
    ap.add_argument("--draft", help="folder under models/ov for speculative decoding")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--draft-device", default=None)
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--assistant", type=int, default=4, help="draft tokens proposed per step")
    a = ap.parse_args()
    path = ROOT / "models" / "ov" / a.model
    cache = ROOT / "models" / "ov_cache"
    kw = {"CACHE_DIR": str(cache)}
    print(f"loading {a.model} on {a.device}" + (f" with draft {a.draft}" if a.draft else ""), flush=True)
    t0 = time.perf_counter()
    if a.draft:
        draft = ov.draft_model(str(ROOT / "models" / "ov" / a.draft), a.draft_device or a.device)
        pipe = ov.LLMPipeline(str(path), a.device, draft_model=draft, **kw)
    else:
        pipe = ov.LLMPipeline(str(path), a.device, **kw)
    print(f"loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    tok = pipe.get_tokenizer()
    prompt = chat_prompt(tok, True)
    run(pipe, prompt, 8, a.assistant if a.draft else None)               # warm up (compiles kernels)
    for i in range(3):
        ttft, tps, text = run(pipe, prompt, a.tokens, a.assistant if a.draft else None, tok)
        print(f"run {i + 1}: first token {ttft * 1000:.0f} ms, {tps:.1f} tokens/s")
    print("TEXT:", text.replace("\n", " ")[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
