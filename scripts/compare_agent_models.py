r"""Which language model for the agent? The same conversations through each one (text, no audio): the action it picks, how long each turn takes,
and the follow-up questions it writes.

  .venv\Scripts\python.exe scripts\compare_agent_models.py openvino_gpu openvino_gpu_8b openvino_gpu_mistral

Only the model changes; the code, the data and the checks are the same. Turns that code decides without the model are marked "code".
Run on AC power with nothing else using the iGPU.
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# (what the caller says, the action that is right; None = any reasonable reply)
CONVERSATIONS = [
    [("I'm looking for a Toyota", "search_cars"), ("something cheaper", "search_cars"), ("the second one", "get_car"),
     ("is it a good family car", None), ("I'd like to test drive it", "prepare_booking"), ("actually never mind, maybe later", None)],
    [("do you have anything with seven seats", "search_cars"), ("what would you recommend for a family of five", None),
     ("tell me about the Land Cruiser", "get_car"), ("is it reliable", None), ("can I see it this weekend", "prepare_booking"),
     ("that's all, thanks, bye", "end_call")],
    [("hi there", "none"), ("I want something sporty", "search_cars"), ("which one is the fastest", None),
     ("the BMW M3", "get_car"), ("can I speak to someone", "handoff")],
]


def run(backend: str) -> dict:
    from dealer_core.app import build_app
    from dealer_core.cli.chat import start
    from dealer_core.config import load_config
    from dealer_core.voice.pipeline import apply_voice_profile

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=Path(r"C:\dev\bmw-portal\site\data.js"), engine="agent", agent_prefix=True, backend=backend)
    start(app, quiet=True)
    apply_voice_profile(app, cfg)
    e = app.engine
    for _ in range(2):                                   # warm: grammar compile, prefix cache
        w = e.new_session()
        e.handle(w, "hello")
    rows = []
    for convo in CONVERSATIONS:
        s = e.new_session()
        e.opening(s)
        for line, want in convo:
            e.prefix_stats = {}
            t0 = time.perf_counter()
            r = e.handle(s, line)
            ms = (time.perf_counter() - t0) * 1000
            by = "code" if "rules" in (r.debug.via or "") else "model"
            rows.append({"line": line, "want": want, "tool": r.debug.intent, "by": by, "ms": round(ms), "reply": r.reply,
                         "follow_up": (e.prefix_stats or {}).get("continuation", "")})
    model_rows = [x for x in rows if x["want"] and x["by"] == "model"]
    out = {"backend": backend, "model": cfg.llm.backends[backend].model,
           "right_actions": sum(x["tool"] == x["want"] for x in model_rows), "model_decided": len(model_rows),
           "turn_ms_median": round(st.median(x["ms"] for x in rows)), "turn_ms_worst": max(x["ms"] for x in rows),
           "follow_ups_accepted": sum(x["follow_up"].startswith("accepted") for x in rows), "follow_ups": sum(bool(x["follow_up"]) and x["follow_up"] != "none-needed" for x in rows),
           "rows": rows}
    return out


def main() -> int:
    backends = sys.argv[1:] or ["openvino_gpu", "openvino_gpu_8b", "openvino_gpu_mistral"]
    results = []
    for b in backends:
        r = run(b)
        results.append(r)
        print(f"\n=== {b} ({r['model']}): right action {r['right_actions']}/{r['model_decided']} of the model's decisions; turn median {r['turn_ms_median']} ms, "
              f"worst {r['turn_ms_worst']} ms; its own follow-up questions accepted {r['follow_ups_accepted']}/{r['follow_ups']}", flush=True)
        for x in r["rows"]:
            mark = "" if not x["want"] or x["tool"] == x["want"] else f"  <- wanted {x['want']}"
            print(f"  {x['ms']:5d} ms [{x['by']:5s} {x['tool']:15s}] {x['line']}\n           {x['reply']}{mark}", flush=True)
    (ROOT / "logs" / "compare_agent_models.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
