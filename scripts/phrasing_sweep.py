r"""Replay many caller phrasings at each step of a call and list what goes wrong (tests/data/phrasing_sweep.yaml).

  uv run python scripts/phrasing_sweep.py                       # code fallback: the model never answers in time (often the case on this laptop)
  uv run python scripts/phrasing_sweep.py --model gpu_server    # the model decides, with its deadline lifted (a stand-in for hardware fast enough to always answer)
  uv run python scripts/phrasing_sweep.py --only time_q --show  # one step, every reply printed

Each case starts a fresh call on a fresh copy of the website's stock (so one case's booking cannot take another's slot), replays the set-up lines to reach its step,
checks the step was reached (a case whose set-up went elsewhere is reported apart, not as a failure), says the phrase, and checks the expectations. Wording is
spoken by code in both modes: this tests understanding (the action, the booking, the details read from the words), not wording. Clock: 21:00, Thu 29 Oct 2026.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NOW = datetime(2026, 10, 29, 21, 0, tzinfo=ZoneInfo("Asia/Dubai"))
SETUP = {                                          # step -> (lines that reach it, a check that it was reached)
    "search": (["I'm looking for a Toyota."], lambda s: bool(s.filters.get("make"))),
    "list": (["I'm looking for a Toyota.", "Any model and any budget is fine."], lambda s: len(s.shown) >= 3 and not s.car),
    "car": (["I'm looking for a Toyota.", "Any model and any budget is fine.", "Tell me about the Corolla."], lambda s: bool(s.car)),
    "day_q": (["I'm looking for a Toyota.", "Any model and any budget is fine.", "Tell me about the Corolla.", "Yes, I'd like a test drive."],
              lambda s: s.state.value == "slot" and not s.offered_slots),
    "time_q": (["I'm looking for a Toyota.", "Any model and any budget is fine.", "Tell me about the Corolla.", "Yes, I'd like a test drive.", "Tomorrow."],
               lambda s: len(s.offered_slots) == 2),
    "contact_q": (["I'm looking for a Toyota.", "Any model and any budget is fine.", "Tell me about the Corolla.", "Yes, I'd like a test drive.", "Tomorrow.",
                   "2 in the afternoon."], lambda s: s.state.value == "contact" and s.draft_slot is not None),
    "name_q": (["I'm looking for a Toyota.", "Any model and any budget is fine.", "Tell me about the Corolla.", "Yes, I'd like a test drive.", "Tomorrow.",
                "2 in the afternoon.", "050 793 2546"], lambda s: s.state.value == "contact" and s.draft_phone is not None and not s.draft_name),
    "readback": (["I'm looking for a Toyota.", "Any model and any budget is fine.", "Tell me about the Corolla.", "Yes, I'd like a test drive.", "Tomorrow.",
                  "2 in the afternoon.", "050 793 2546", "Fatima"], lambda s: s.read_back),
    "booked": (["I'm looking for a Toyota.", "Any model and any budget is fine.", "Tell me about the Corolla.", "Yes, I'd like a test drive.", "Tomorrow.",
                "2 in the afternoon.", "050 793 2546", "Fatima", "yes"], lambda s: s.booking is not None),
}


class TooSlow:
    """A model that never answers in time: every turn the rules do not route is decided by the code's fallback, and every reply is the code's wording."""

    def lead_in(self, *a, **k):
        from dealer_core.nlu.backend import BackendTimeout
        raise BackendTimeout("too slow")

    def lead_in_stream(self, *a, **k):
        from dealer_core.nlu.backend import BackendTimeout
        raise BackendTimeout("too slow")
        yield ""


class Stub:
    name = "stub"

    def classify(self, *a, **k):
        from dealer_core.dialogue.states import Intent
        from dealer_core.nlu.backend import IntentResult
        return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")


def make_engine(cfg, portal_data: Path, decider):
    from dealer_core.agent.engine import AgentEngine
    from dealer_core.agent.governor import QualityGovernor
    from dealer_core.app import build_app
    from dealer_core.dialogue.writer import ReplyWriter

    app = build_app(cfg, inventory="portal", portal_data=portal_data, classifier=Stub(), reply_mode="conversational", clock=lambda: NOW.astimezone(timezone.utc),
                    lead_backend=TooSlow())
    e = AgentEngine(app.service, decider, ReplyWriter(TooSlow(), temperature=None))
    e.governor = QualityGovernor("code")                                    # the code's wording at once: only the decisions are under test
    return e


def check(expect: dict, intent: str, reply: str, s, e) -> list[str]:
    out = []
    low = reply.lower()
    car = e.svc.get_car(s.car) if s.car else None
    booked = s.booking is not None
    name = (s.booking.customer_name if booked else s.draft_name) or ""
    slot = s.booking.start if booked else s.draft_slot
    phone = (s.booking.customer_phone if booked else s.draft_phone) or ""
    for key, want in expect.items():
        if key == "tool" and intent not in ([want] if isinstance(want, str) else want):
            out.append(f"action {intent}, wanted {want}")
        elif key == "not_tool" and intent in ([want] if isinstance(want, str) else want):
            out.append(f"action {intent}, which it must not be")
        elif key == "state" and s.state.value not in ([want] if isinstance(want, str) else want):
            out.append(f"state {s.state.value}, wanted {want}")
        elif key == "booked" and booked != want:
            out.append("booked" if booked else "not booked")
        elif key == "name" and name.lower() != str(want).lower():
            out.append(f"name {name or 'none'!r}, wanted {want!r}")
        elif key == "slot_hour" and (slot is None or slot.hour != want):
            out.append(f"time {slot.strftime('%a %H:%M') if slot else 'none'}, wanted {want}:00")
        elif key == "phone" and not phone.endswith(str(want)):
            out.append(f"phone {phone or 'none'}, wanted ...{want}")
        elif key == "car" and (car is None or str(want).lower() not in car.model.lower()):
            out.append(f"car {car.model if car else 'none'}, wanted {want}")
        elif key == "keeps_booking" and want and not (s.state.value in ("slot", "contact", "confirm") or s.read_back or booked):
            out.append(f"booking talk dropped (state {s.state.value})")
        elif key == "reply_has" and not any(w.lower() in low for w in want):
            out.append(f"reply has none of {want}")
        elif key == "reply_lacks" and any(w.lower() in low for w in want):
            out.append(f"reply has {[w for w in want if w.lower() in low]}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="", help="a backend from config (e.g. gpu_server) to make the decisions, deadline lifted; default: the code fallback")
    ap.add_argument("--cases", type=Path, default=ROOT / "tests" / "data" / "phrasing_sweep.yaml")
    ap.add_argument("--portal-data", type=Path, default=ROOT.parent / "bmw-portal" / "site" / "data.js")
    ap.add_argument("--only", default="", help="only the cases at this step")
    ap.add_argument("--show", action="store_true", help="print every case's reply")
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "phrasing_sweep.json")
    a = ap.parse_args()

    from dealer_core.config import load_config

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    decider = TooSlow()
    if a.model:
        from dealer_core.nlu.chain import build_backend
        decider = build_backend(a.model, cfg)
    cases = [c for c in yaml.safe_load(a.cases.read_text(encoding="utf-8")) if not a.only or c["at"] == a.only]
    results, t0 = [], time.perf_counter()
    for i, case in enumerate(cases):
        e = make_engine(cfg, a.portal_data, decider)
        if a.model:
            e.decide_deadline_s = 8.0                                       # the model always answers: what it decides, not whether it is in time
        s = e.new_session()
        lines, reached = SETUP[case["at"]]
        for line in lines:
            list(e.handle_stream(s, line))
        if not reached(s):
            results.append({**case, "result": "setup", "why": [f"set-up for {case['at']} went elsewhere (state {s.state.value})"]})
            continue
        out = list(e.handle_stream(s, case["say"]))
        intent = out[-1].debug.intent
        reply = " ".join(x for x in out if isinstance(x, str))
        why = check(case["expect"], intent, reply, s, e)
        results.append({**case, "result": "fail" if why else "pass", "why": why, "intent": intent, "reply": reply, "via": out[-1].debug.via})
        if a.show or why:
            print(f"[{'FAIL' if why else 'ok  '}] {case['at']:9s} {case['say']!r:46s} {intent:15s} {'; '.join(why)}\n{'':18s}-> {reply[:150]}", flush=True)
    counts = Counter(r["result"] for r in results)
    by_step = {}
    for r in results:
        st = by_step.setdefault(r["at"], Counter())
        st[r["result"]] += 1
    print(f"\n{'model ' + a.model + ' (deadline lifted)' if a.model else 'code fallback (the model never answers in time)'}: {counts['pass']} passed, {counts['fail']} failed, "
          f"{counts['setup']} set-up went elsewhere, of {len(results)} ({time.perf_counter() - t0:.0f} s)")
    for step, c in by_step.items():
        print(f"  {step:10s} {c['pass']:3d} passed  {c['fail']:3d} failed" + (f"  {c['setup']} set-up" if c["setup"] else ""))
    a.out.parent.mkdir(exist_ok=True)
    a.out.write_text(json.dumps({"mode": a.model or "code", "counts": dict(counts), "results": results}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
