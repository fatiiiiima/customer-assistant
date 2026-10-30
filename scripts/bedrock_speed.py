r"""How fast is each model on Amazon Bedrock at the two things the voice agent asks a model for? (30 Oct, the owner's Bedrock key.)

1. Decide: name the next action ("get_car S1042", "search_cars"). This is on the way to the first sound of a reply, so it is what the
   1 s rule depends on. ARCH_PLAN wants the model's part under 300 ms.
2. Follow-up: the closing question after a reply the code has begun ("prefix" mode). This is spoken after the code's words, so it is less tight.

Two steps:

  uv run python scripts/bedrock_speed.py capture
      Runs the conversations of compare_agent_models.py through the agent on the LOCAL model server (127.0.0.1:8003, free) and saves every
      prompt the agent sent to benchmarks/bedrock_speed_prompts.json. The cars are the website's demo cars (bmw-portal's data.js), as on the
      live demo; the conversations hold no real caller.

  uv run python scripts/bedrock_speed.py run --dry-run      what would be sent, how many calls, about how many tokens; nothing is sent
  BEDROCK_API_KEY=... VOICE_AGENT_ALLOW_CLOUD=1 uv run python scripts/bedrock_speed.py run
      Sends those prompts, word for word, to each model: 1 warm-up call, then 10 decisions and 10 follow-ups (21 calls a model, 84 for four).
      This COSTS MONEY. The key is read from BEDROCK_API_KEY and never printed or saved.

Results: a table here and benchmarks/results/bedrock_speed_<date>.json (ignored by git). Times are measured from this machine, so they include
the network trip to the region; `network` is the time of a plain GET /models, the floor under every call.

The agent's client sends a few fields only vLLM knows (continue_final_message, structured_outputs, chat_template_kwargs). If a model refuses
the warm-up with HTTP 400, it is tried again with standard OpenAI fields only ("plain"); the table says which way worked.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics as st
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

KEY_ENV, ALLOW_ENV = "BEDROCK_API_KEY", "VOICE_AGENT_ALLOW_CLOUD"
BASE = "https://bedrock-mantle.us-east-1.api.aws/v1"
MODELS = ["qwen.qwen3-next-80b-a3b-instruct", "mistral.ministral-3-8b-instruct", "anthropic.claude-haiku-4-5", "google.gemma-4-26b-a4b"]
PROMPTS = ROOT / "benchmarks" / "bedrock_speed_prompts.json"
SITE_DATA = ROOT.parent / "bmw-portal" / "site" / "data.js"
LOCAL = {"VA__LLM__BACKENDS__GPU_SERVER__HOST": "http://127.0.0.1:8003/v1", "VA__LLM__BACKENDS__GPU_SERVER__MODEL": "mlx-community/Qwen3-4B-Instruct-2507-4bit"}
VLLM_ONLY = ("continue_final_message", "add_generation_prompt", "guided_json", "structured_outputs", "chat_template_kwargs")
TARGET_MS = 300                      # ARCH_PLAN: the model's part of a reply


# ------------------------------------------------------------------------------------------------ capture (local, free)
def capture(out: Path) -> int:
    from compare_agent_models import CONVERSATIONS
    from dealer_core.agent.engine import DECIDE_SYSTEM
    from dealer_core.app import build_app
    from dealer_core.cli.chat import start
    from dealer_core.config import load_config

    cfg = load_config(environ={**LOCAL, "VA__DIALOGUE__DECISION_BACKEND": "gpu_server"}, overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=SITE_DATA, backend="gpu_server", engine="agent", quality="prefix")
    start(app, quiet=True)
    e, records, now = app.engine, [], {"line": "", "want": None}
    decider, writer = e.backend, e.reply_writer.backend
    real_choose, real_lead_in, real_stream = getattr(decider, "choose", None), decider.lead_in, writer.lead_in_stream

    def note_decide(system, user, options, max_tokens, deadline_s):
        if system == DECIDE_SYSTEM and not (records and records[-1]["kind"] == "decide" and records[-1]["user"] == user):   # choose() refused, then lead_in(): one decision
            records.append({"kind": "decide", "system": system, "user": user, "options": list(options or []), "max_tokens": max_tokens,
                            "customer": now["line"], "want": now["want"]})

    def choose(system, user, options, max_tokens=16, deadline_s=None):
        note_decide(system, user, options, max_tokens, deadline_s)
        return real_choose(system, user, options, max_tokens, deadline_s)

    def lead_in(system, user, max_tokens=36, temperature=None, deadline_s=None):
        note_decide(system, user, None, max_tokens, deadline_s)
        return real_lead_in(system, user, max_tokens, temperature, deadline_s)

    def stream(system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        if prefill:
            records.append({"kind": "follow_up", "system": system, "user": user, "max_tokens": max_tokens, "temperature": temperature,
                            "prefill": prefill, "customer": now["line"]})
        yield from real_stream(system, user, max_tokens, temperature, deadline_s, prefill=prefill)

    if real_choose is not None:
        decider.choose = choose
    decider.lead_in, writer.lead_in_stream = lead_in, stream
    for convo in CONVERSATIONS:
        s = e.new_session()
        e.opening(s)
        for line, want in convo:
            now.update(line=line, want=want)
            e._decide(s, line)                     # the decision the model would be asked here even when code decides it: 2 of 17 turns reach the model,
            r = e.handle(s, line)                  # and 2 prompts sent ten times over would flatter a server that caches what it has just seen
            print(f"  [{r.debug.intent:15s} {r.debug.via:12s}] {line}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    kinds = [r["kind"] for r in records]
    print(f"saved {kinds.count('decide')} decisions and {kinds.count('follow_up')} follow-ups to {out.relative_to(ROOT)}")
    return 0


# ------------------------------------------------------------------------------------------------ run (Bedrock, costs money)
def make_backend(model: str, base: str, plain: bool = False):
    """The agent's own client for an OpenAI-compatible server, pointed at Bedrock, counting what it does."""
    from dealer_core.config import BackendCfg, load_config
    from dealer_core.nlu.openai_compat import OpenAICompatBackend

    class Timed(OpenAICompatBackend):
        def __init__(self, *a):
            super().__init__(*a)
            self.plain, self.streams, self.usage = plain, 0, {"prompt_tokens": 0, "completion_tokens": 0}

        def _body(self, system, user, max_tokens, temperature, prefill="", schema=None, stream=False):
            if self.plain and prefill:                     # no "continue this reply" on a standard server: say what was already said
                user, prefill = f"{user}\n(You have already said: \"{prefill.strip()}\". Write only what comes next.)", ""
            body = super()._body(system, user, max_tokens, temperature, prefill, schema, stream)
            for k in VLLM_ONLY if self.plain else ():
                body.pop(k, None)
            return body

        def _post(self, body, timeout):
            out = super()._post(body, timeout)
            for k in self.usage:
                self.usage[k] += (out.get("usage") or {}).get(k) or 0
            return out

        def _stream(self, body, deadline_s):
            self.streams += 1                              # two for one follow-up: the server refused to continue a begun reply
            yield from super()._stream(body, deadline_s)

        def choose(self, *a, **k):
            return None if self.plain else super().choose(*a, **k)

    llm = load_config(environ={}).llm
    return Timed(f"bedrock:{model}", BackendCfg(runtime="openai", host=base, model=model, timeout_s=10.0, api_key_env=KEY_ENV), llm)


def decide(b, rec: dict) -> str:
    """As the agent does it (engine._decide): the listed actions if the server can be held to them, otherwise free text."""
    raw = b.choose(rec["system"], rec["user"], rec["options"], rec["max_tokens"], 10.0) if rec["options"] else None
    if raw is None:
        raw = b.lead_in(rec["system"], rec["user"], rec["max_tokens"], None, 10.0)
    return re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()


def tool_of(raw: str) -> str:
    m = re.search(r'"tool"\s*:\s*"([a-z_]+)"', raw)
    return m.group(1) if m else (raw.split() or [""])[0].strip('"{},.')


def follow_up(b, rec: dict) -> tuple[float | None, str]:
    """Time to the end of the first sentence (where the agent stops reading), and the sentence."""
    t0, buf = time.perf_counter(), ""
    for piece in b.lead_in_stream(rec["system"], rec["user"], rec["max_tokens"], rec["temperature"], 10.0, prefill=rec["prefill"]):
        buf += piece
        m = re.search(r"[.!?](?:\s|$)", buf)
        if m:
            return (time.perf_counter() - t0) * 1000, buf[:m.end()].strip()
    return ((time.perf_counter() - t0) * 1000 if buf.strip() else None), buf.strip()


def pct(xs: list[float], p: float) -> float | None:
    xs = sorted(xs)
    return round(xs[max(0, math.ceil(p / 100 * len(xs)) - 1)]) if xs else None


def short(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"[:200]


def one_model(model: str, base: str, records: list[dict], n_decide: int, n_follow: int, say=print) -> dict:
    decisions = [r for r in records if r["kind"] == "decide"]
    follows = [r for r in records if r["kind"] == "follow_up"]
    out: dict = {"model": model, "mode": "agent", "network_ms": [], "decide": [], "follow_up": [], "errors": []}
    b = make_backend(model, base)
    h = b.health()
    out["health"] = h.detail if hasattr(h, "detail") else str(h)
    for _ in range(3):
        t0 = time.perf_counter()
        try:
            b.client.get(f"{b.base}/models", timeout=10.0)
            out["network_ms"].append(round((time.perf_counter() - t0) * 1000))
        except Exception as e:
            out["errors"].append(f"network: {short(e)}")
    for plain in (False, True):                                       # the warm-up, not counted
        if plain:
            b, out["mode"] = make_backend(model, base, plain=True), "plain"
        try:
            decide(b, decisions[0])
            break
        except Exception as e:
            out["errors"].append(f"warm-up ({out['mode']}): {short(e)}")
            if "HTTP 400" not in str(e) or plain:
                say(f"  {model}: the warm-up failed ({short(e)}); skipped")
                return out
    for rec in (decisions * n_decide)[:n_decide]:
        t0 = time.perf_counter()
        try:
            raw = decide(b, rec)
            tool = tool_of(raw)
            out["decide"].append({"ms": round((time.perf_counter() - t0) * 1000), "customer": rec["customer"], "raw": raw[:80],
                                  "listed": raw in rec["options"], "tool": tool, "want": rec.get("want"),
                                  "right": None if not rec.get("want") else tool == rec["want"]})
        except Exception as e:
            out["errors"].append(f"decide: {short(e)}")
    streams0 = b.streams
    for rec in (follows * n_follow)[:n_follow]:
        try:
            ms, sentence = follow_up(b, rec)
            out["follow_up"].append({"ms": None if ms is None else round(ms), "customer": rec["customer"], "said_first": rec["prefill"][-60:], "sentence": sentence})
        except Exception as e:
            out["errors"].append(f"follow-up: {short(e)}")
    out["prefill_refused"] = b.streams - streams0 > len(out["follow_up"]) + sum(x.startswith("follow-up") for x in out["errors"])
    out["constrained_choice"] = bool(getattr(b, "can_choose", False)) and not b.plain
    out["usage_non_streamed"] = b.usage
    return out


def summary(r: dict) -> dict:
    d = [x["ms"] for x in r["decide"]]
    f = [x["ms"] for x in r["follow_up"] if x["ms"] is not None]
    judged = [x for x in r["decide"] if x["right"] is not None]
    return {"model": r["model"], "mode": r["mode"], "network_p50": pct(r["network_ms"], 50),
            "decide_p50": pct(d, 50), "decide_p90": pct(d, 90), "decide_n": len(d),
            "right": f"{sum(x['right'] for x in judged)}/{len(judged)}" if judged else "-",
            "follow_p50": pct(f, 50), "follow_p90": pct(f, 90), "follow_n": len(f), "errors": len(r["errors"]),
            "decide_under_target": None if not d else pct(d, 90) < TARGET_MS}


def table(rows: list[dict]) -> str:
    head = f"{'model':36s} {'mode':6s} {'network':>8s} {'decide p50/p90':>15s} {'right':>6s} {'follow-up p50/p90':>18s} {'errors':>6s}"
    lines = [head, "-" * len(head)]
    for s in rows:
        ms = lambda a, b: f"{a if a is not None else '-'}/{b if b is not None else '-'}"
        lines.append(f"{s['model']:36s} {s['mode']:6s} {str(s['network_p50'] or '-'):>8s} {ms(s['decide_p50'], s['decide_p90']):>15s} {s['right']:>6s} "
                     f"{ms(s['follow_p50'], s['follow_p90']):>18s} {s['errors']:>6d}")
    return "\n".join(lines)


def run(a) -> int:
    if not a.prompts.exists():
        print(f"no prompts at {a.prompts}: run `scripts/bedrock_speed.py capture` first (local model server, free)")
        return 2
    records = json.loads(a.prompts.read_text(encoding="utf-8"))
    n_dec, n_fol = sum(r["kind"] == "decide" for r in records), sum(r["kind"] == "follow_up" for r in records)
    if not n_dec or (a.follow_ups and not n_fol):
        print(f"{a.prompts.name} has {n_dec} decisions and {n_fol} follow-ups; capture again")
        return 2
    calls = 1 + a.decisions + a.follow_ups
    chars = lambda kind, n: sum(len(r["system"]) + len(r["user"]) for r in ([x for x in records if x["kind"] == kind] * n)[:n])
    est_in = (chars("decide", a.decisions + 1) + chars("follow_up", a.follow_ups)) // 4
    print(f"{len(a.models)} model(s) x {calls} calls = {len(a.models) * calls} calls to {a.base}\n"
          f"about {est_in:,} input tokens and at most {16 * (a.decisions + 1) + 36 * a.follow_ups:,} output tokens a model (input: characters / 4, an estimate)")
    if a.dry_run:
        print("dry run: nothing sent\n" + "\n".join(f"  {m}" for m in a.models))
        return 0
    if not os.environ.get(KEY_ENV) or os.environ.get(ALLOW_ENV) != "1":
        print(f"not sent: set {KEY_ENV} (your key) and {ALLOW_ENV}=1 in your own terminal first. This run costs money.")
        return 2
    results = []
    for m in a.models:
        print(f"{m} ...", flush=True)
        results.append(one_model(m, a.base, records, a.decisions, a.follow_ups))
    rows = [summary(r) for r in results]
    print("\nmilliseconds, from this machine (network = a plain GET /models)\n" + table(rows))
    for r in results:
        for e in r["errors"][:3]:
            print(f"  {r['model']}: {e}")
        for x in r["follow_up"][:2]:
            print(f"  {r['model']} follow-up: \"{x['said_first']}\" -> \"{x['sentence']}\"")
    out = ROOT / "benchmarks" / "results" / f"bedrock_speed_{date.today().isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"base": a.base, "summary": rows, "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {out.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="record the agent's prompts on the local model server (free)")
    c.add_argument("--out", type=Path, default=PROMPTS)
    r = sub.add_parser("run", help="send them to Bedrock (costs money; --dry-run sends nothing)")
    r.add_argument("--models", nargs="+", default=MODELS)
    r.add_argument("--base", default=BASE, help="the Bedrock OpenAI-compatible endpoint (another region: change us-east-1)")
    r.add_argument("--decisions", type=int, default=10)
    r.add_argument("--follow-ups", type=int, default=10)
    r.add_argument("--prompts", type=Path, default=PROMPTS)
    r.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return capture(a.out) if a.cmd == "capture" else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
