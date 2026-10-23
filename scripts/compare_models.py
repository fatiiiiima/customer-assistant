r"""Replay the SAME prompts (recorded by scripts/capture_prompts.py) on several local models and compare speed, safety-check pass rate, tool choice and wording.

  .venv\Scripts\python.exe scripts\compare_models.py logs\prompts.json --ov Qwen3-4B-Instruct-2507-int4-ov-nocal Phi-4-mini-instruct-int4-ov --ollama gemma3:4b

For every recorded reply prompt: time to first token, tokens/s, how many of the model's sentences the writer's safety checks accept, and the text.
For every recorded decision prompt: the tool the model names, against the tool the reference model named.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TOOLS = ("search_cars", "get_car", "check_slots", "prepare_booking", "confirm_booking", "cancel_draft", "handoff", "end_call", "none")


def parts(user: str) -> tuple[str, str, str]:
    """(facts, core, caller) out of a reply prompt written by ReplyWriter.stream."""
    facts = user.split("FACTS FROM THE SYSTEM:\n", 1)[1].split("\n\nSTANDARD WORDING", 1)[0] if "FACTS FROM THE SYSTEM:" in user else ""
    core = user.split("STANDARD WORDING (reference only, do not copy): ", 1)[1].split("\n\n", 1)[0] if "STANDARD WORDING" in user else ""
    caller = user.split("CUSTOMER (just now): ", 1)[1].split("\n\n", 1)[0] if "CUSTOMER (just now):" in user else ""
    return facts, core, caller


class OvModel:
    def __init__(self, folder: str, device: str = "GPU") -> None:
        import openvino_genai as ov

        self.ov, self.name = ov, folder
        self.pipe = ov.LLMPipeline(str(ROOT / "models" / "ov" / folder), device, CACHE_DIR=str(ROOT / "models" / "ov_cache"))
        self.tok = self.pipe.get_tokenizer()
        self.hybrid_qwen3 = folder.startswith("Qwen3-") and "2507" not in folder

    def run(self, system: str, user: str, max_tokens: int) -> tuple[str, float, float, int]:
        ov = self.ov
        u = user + (" /no_think" if self.hybrid_qwen3 else "")
        prompt = self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": u}], add_generation_prompt=True,
                                              extra_context={"enable_thinking": False})
        cfg = ov.GenerationConfig()
        cfg.max_new_tokens, cfg.do_sample = max_tokens, False
        first: list[float] = []
        t0 = time.perf_counter()

        def streamer(_piece: str):
            if not first:
                first.append(time.perf_counter() - t0)
            return ov.StreamingStatus.RUNNING

        out = str(self.pipe.generate(prompt, cfg, streamer))
        total = time.perf_counter() - t0
        out = re.sub(r"^\s*<think>.*?</think>\s*", "", out, flags=re.S).strip()
        n = len(self.tok.encode(out).input_ids.data[0])
        return out, (first[0] if first else total), total, n


class OllamaModel:
    def __init__(self, tag: str) -> None:
        self.name = "ollama:" + tag
        self.tag = tag

    def run(self, system: str, user: str, max_tokens: int) -> tuple[str, float, float, int]:
        import requests

        t0 = time.perf_counter()
        first: list[float] = []
        text, n = "", 0
        r = requests.post("http://localhost:11434/api/chat", stream=True, timeout=120, json={
            "model": self.tag, "stream": True, "options": {"temperature": 0, "num_predict": max_tokens},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        for line in r.iter_lines():
            if not line:
                continue
            d = json.loads(line)
            piece = (d.get("message") or {}).get("content", "")
            if piece:
                first = first or [time.perf_counter() - t0]
                text += piece
                n += 1
        total = time.perf_counter() - t0
        return text.strip(), (first[0] if first else total), total, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts")
    ap.add_argument("--ov", nargs="*", default=[], help="folders under models/ov")
    ap.add_argument("--ollama", nargs="*", default=[], help="Ollama model tags")
    a = ap.parse_args()
    from dealer_core.dialogue import voice_rules
    from dealer_core.dialogue.writer import verify_sentence

    records = json.loads(Path(a.prompts).read_text(encoding="utf-8"))
    models = []
    for f in a.ov:
        print(f"loading {f} ...", flush=True)
        models.append(OvModel(f))
    models += [OllamaModel(t) for t in a.ollama]
    summary = []
    for m in models:
        m.run("Say hi.", "hi", 4)                                              # warm up
        ttft, tps, ok, total_sent, agree, n_dec, rows = [], [], 0, 0, 0, 0, []
        for r in records:
            if r["kind"] == "decide":
                out, t1, tt, n = m.run(r["system"], r["user"], 14)
                tool = next((t for t in TOOLS if re.search(r"\b" + t + r"\b", out)), "none")
                ref = next((t for t in TOOLS if re.search(r"\b" + t + r"\b", r["tool"])), "none")
                n_dec += 1
                agree += tool == ref
                rows.append(f"  DECIDE {r['customer']!r}: {tool}" + ("" if tool == ref else f"   (reference: {ref})"))
                continue
            out, t1, tt, n = m.run(r["system"], r["user"], r.get("max_tokens", 70))
            facts, core, caller = parts(r["user"])
            sents = voice_rules.sentences(out)
            bad = [s for s in sents if verify_sentence(s, caller, core, facts, "")]
            ok += len(sents) - len(bad)
            total_sent += len(sents)
            ttft.append(t1)
            tps.append((n - 1) / max(tt - t1, 1e-6) if n > 1 else 0)
            rows.append(f"  REPLY {r['customer']!r}: {out}" + (f"\n        REJECTED by the checks: {bad}" if bad else ""))
        avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
        summary.append((m.name, avg(ttft) * 1000, avg(tps), ok, total_sent, agree, n_dec))
        print(f"\n=== {m.name}\n" + "\n".join(rows), flush=True)
    print("\nSUMMARY (same prompts for every model)")
    print(f"{'model':45s} {'first token':>11s} {'tok/s':>6s}  {'sentences passing checks':>24s}  {'tool = reference':>16s}")
    for name, t, v, ok, tot, ag, nd in summary:
        print(f"{name:45s} {t:9.0f}ms {v:6.1f}  {ok:>11d}/{tot:<3d} ({100 * ok / max(tot, 1):3.0f}%)  {ag:>9d}/{nd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
