r"""Type-to-chat CLI for the dealership agent (text only; the same engine and tools the voice agent will use).

  .venv\Scripts\python.exe -m dealer_core.cli.chat --backend openvino_gpu --debug

Commands while chatting:  /debug (toggle trace)  /state (show the session)  /reset (new call)  /quit
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from ..app import App, build_app, demo_clock
from ..config import load_config
from ..dialogue.machine import TurnDebug
from ..nlu.backend import BackendError


def format_debug(d: TurnDebug) -> str:
    """The per-turn trace: state, intent, who handled it (rules or LLM), tools, policy decision, timings."""
    who = {"rules": "RULES", "llm": "LLM", "language": "LANGUAGE CHECK"}.get(d.via.split(":")[0], d.via)
    detail = d.via.split(":", 1)[1] if ":" in d.via else ""
    lines = [f"  | state   : {d.state_before} -> {d.state_after}",
             f"  | intent  : {d.intent}   handled by {who}" + (f" ({detail})" if detail else "") + f"   NLU {d.nlu_ms:.1f} ms",
             f"  | tool    : {'; '.join(d.tools) if d.tools else '-'}",
             f"  | policy  : {d.decision or '-'}"]
    if d.draft:
        lines.append(f"  | draft   : model-written full reply, verified against the facts ({d.lead_ms:.0f} ms)")
    if d.lead_in:
        lines.append(f"  | lead-in : \"{d.lead_in}\"   (model-written, verified; {d.lead_ms:.0f} ms)")
    if d.note:
        lines.append(f"  | note    : {d.note}")
    if d.violations:
        lines.append(f"  | VOICE-RULE VIOLATIONS: {d.violations}")
    lines.append(f"  | time    : {d.total_ms:.1f} ms total")
    return "\n".join(lines)


def start(app: App, quiet: bool = False) -> None:
    """Health-check the NLU backends (loads the model; takes a few seconds the first time) and say which one is active."""
    if app.chain is None:
        return
    t0 = time.perf_counter()
    if not quiet:
        print("Loading the language model (first start takes a few seconds)...", flush=True)
    msg = app.chain.start()
    if not quiet:
        print(f"{msg}  [{time.perf_counter() - t0:.1f} s]\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", help="llm backend from config (openvino_gpu | openvino_gpu_small | openvino_cpu | ollama); default llm.text_backend")
    ap.add_argument("--inventory", choices=["demo", "seed"], default="demo", help="demo = 16 hand-written fake cars; seed = 60 random fake cars")
    ap.add_argument("--clock", choices=["demo", "real"], default="demo", help="demo = frozen at the next Monday 10:00 (scripted demos look the same any day)")
    ap.add_argument("--no-draft", action="store_true", help="keep the lead-ins but do not let the model rewrite whole short replies")
    ap.add_argument("--template-only", action="store_true", help="switch the hybrid lead-ins off (templates only, fastest, fully fixed wording)")
    ap.add_argument("--hybrid", action="store_true", help="hybrid replies: a verified model-written lead-in sentence before low-risk templated answers")
    ap.add_argument("--debug", action="store_true", help="show state, intent, rules-or-LLM, tools, policy decision and timing for every turn")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    app = build_app(cfg, inventory=a.inventory, backend=a.backend, clock=demo_clock(cfg) if a.clock == "demo" else None,
                    lead_in=False if a.template_only else (True if a.hybrid else None),
                    draft=False if (a.template_only or a.no_draft) else None)
    try:
        start(app)
    except BackendError as e:
        print(f"No language-model backend is available ({e}).\nRules-only mode: the agent will hand off whenever the rules cannot decide.\n")
    debug = a.debug
    s = app.engine.new_session()
    print("(demo data: all cars, prices and customers are fake. Type /quit to leave, /debug to toggle the trace.)\n")
    print(f"Agent: {app.engine.opening(s)}")
    while True:
        try:
            text = input("\nYou:   ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text == "/quit":
            return 0
        if text == "/debug":
            debug = not debug
            print(f"(debug trace {'on' if debug else 'off'})")
            continue
        if text == "/reset":
            s = app.engine.new_session()
            print(f"(new call)\nAgent: {app.engine.opening(s)}")
            continue
        if text == "/state":
            print(f"(state={s.state.value} car={s.car} shown={[c.stock_id for c in s.shown]} filters={s.filters} pending={s.pending} "
                  f"booking={s.booking.booking_id if s.booking else None})")
            continue
        r = app.engine.handle(s, text)
        print(f"Agent: {r.reply}")
        if debug:
            print(format_debug(r.debug))
        if r.ended:
            why = s.handoff.reason if s.handoff else "call ended"
            print(f"\n[{'transferred to a human (' + why + ')' if s.handoff else 'call ended'}]  Type /reset for a new call or /quit to leave.")


if __name__ == "__main__":
    sys.exit(main())
