r"""Conversations replayed through the real agent engine with a real model at the FULL level, with `dialogue.model_speaks` on or off (ARCH_PLAN Phase 3).

  uv run python scripts/replay_model_speaks.py --backend gpu_server                    # model_speaks on, every conversation once
  uv run python scripts/replay_model_speaks.py --backend gpu_server --off              # the full level as before, for comparison
  uv run python scripts/replay_model_speaks.py --backend gpu_server --turns 1000        # repeat the conversations until 1,000 model-written replies were checked

Conversations: the five scripted ones of scripts/load_test.py (three of them book a test drive), a "tell me more about X" script, and the words the caller was heard
to say in every recorded call (logs/audio_dump/*_in.txt, "heard" lines). Text only: no audio, no recogniser. The clock is fixed (Tuesday 27 Oct 2026, 11:00 Dubai),
so the showroom is open and bookings can be made.

Per run: model-written replies, how many fell back to the code's wording, sentences the fact guard (agent/guards.py) stopped, sentences SPOKEN that still say a
number, colour, car name or commercial term not in that turn's facts (re-checked after the fact; should be 0), replies that end on a question, replies that open with
the same three words as an earlier reply in the call, fact sentences said twice in a call, "tell me more about X" answered with X, bookings made, and the model's time
to its first sentence (THIS machine's time, not a production number).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

NOW = datetime(2026, 10, 27, 11, 0, tzinfo=ZoneInfo("Asia/Dubai"))
TELL_ME = [("Porsche.", None), ("Yes, tell me about the Macan.", None), ("Tell me more about the engine.", "horsepower"), ("And the boot?", "boot"),
           ("Tell me more about its history.", "history"), ("What about the fuel economy?", "litres"), ("How many owners has it had?", "owner"),
           ("Tell me more about the interior.", "interior")]
BOOKS = {0, 1, 3}
TEST_CALL = ["I'm looking for a BMW.", "Any budget, any model?", "I want to hear more about the BMW X5.", "Tell me more about it.", "I want to see the car.",
             "I want to see the car, I want to go on a test drive.", "Any day works.", "9 in the morning.", "Fatima 050, 793, 2546", "yes, please book it"]   # the owner's 28 Oct call
# A rough diagnostic, not a check: phrases that judge or embellish rather than state ("known for strong performance", "practical for weekend drives"). The fact checks
# do not catch these (no number, name or commercial term in them); the prompt forbids praise. Counted so a listening test knows where to look.
PRAISE = re.compile(r"\b(?:known for|perfect for|ideal for|great for|enough for|making it|makes it|strong|responsive|smooth|comfortable|spacious|luxurious|sporty|"
                    r"reliable|impressive|excellent|fantastic|amazing|stunning|beautiful|solid|great shape|well looked after|popular|practical for)\b", re.I)                                  # load_test.MIX conversations that end in a confirmed booking


def heard_lines(folder: Path) -> list[list[str]]:
    calls = []
    for p in sorted(folder.glob("*_in.txt")):
        lines = [ln.split("\t")[2] for ln in p.read_text(encoding="utf-8").splitlines() if ln.count("\t") >= 2 and ln.split("\t")[1] == "heard"]
        if len(lines) >= 2:
            calls.append(lines)
    return calls


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="gpu_server")
    ap.add_argument("--off", action="store_true", help="dialogue.model_speaks off: the full level as it was")
    ap.add_argument("--turns", type=int, default=0, help="repeat the conversations until this many model-written replies were checked (0 = each once)")
    ap.add_argument("--portal-data", type=Path, default=ROOT.parent / "bmw-portal" / "site" / "data.js")
    ap.add_argument("--no-deadline", action="store_true", help="judge correctness, not speed: no first-sentence deadline and a 10 s reply deadline (on this laptop the full prompt "
                    "takes longer than the live 1.5 s, so with the deadline every reply falls back to the code's wording). The time to the first sentence is still recorded")
    ap.add_argument("--reply-deadline-s", type=float, default=None, help="with --no-deadline: the whole-reply deadline instead of 10 s (the owner's test call ran with 4.5 s)")
    ap.add_argument("--show", action="store_true", help="print every turn")
    ap.add_argument("--only", default="", help="run only the conversations whose name starts with this (mix, tell-me-more, call)")
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "replay_model_speaks.json")
    a = ap.parse_args()

    from load_test import MIX

    from dealer_core.agent import guards
    from dealer_core.app import build_app
    from dealer_core.cli.chat import start
    from dealer_core.config import load_config
    from dealer_core.dialogue import voice_rules
    from dealer_core.voice.pipeline import apply_voice_profile

    cfg = load_config(overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=a.portal_data, engine="agent", backend=a.backend, quality="full", clock=lambda: NOW.astimezone(timezone.utc))
    start(app, quiet=True)
    apply_voice_profile(app, cfg)
    e = app.engine
    e.model_speaks = not a.off
    if a.no_deadline and e.reply_writer is not None:
        e.reply_writer.first_deadline_s, e.reply_writer.deadline_s = None, a.reply_deadline_s or 10.0
    names = guards.car_names(app.inventory.vocabulary())

    seen: dict = {}
    real_input = e._speaking_input

    def recording_input(s, text, result):
        core, awaiting, facts = real_input(s, text, result)
        seen["allowed"] = f"{facts}\n{core}\n{text}"
        return core, awaiting, facts

    e._speaking_input = recording_input
    conversations = [("mix%d" % i, [(t, None) for t in c]) for i, c in enumerate(MIX)] + [("tell-me-more", TELL_ME), ("testcall", [(t, None) for t in TEST_CALL])] + \
                    [("call%02d" % i, [(t, None) for t in c]) for i, c in enumerate(heard_lines(ROOT / "logs" / "audio_dump"))]
    conversations = [c for c in conversations if c[0].startswith(a.only)]
    stats = {"model_speaks": not a.off, "deadlines": "off (correctness run)" if a.no_deadline else "live", "conversations": 0, "turns": 0, "model_replies": 0, "fell_back": 0, "guard_stops": 0, "spoken_leaks": 0, "ends_on_question": 0,
             "same_opener": 0, "repeated_fact_sentences": 0, "tell_me_more": [0, 0], "bookings": {}, "first_ms": [], "leak_examples": [], "fallback_reasons": {}, "praise_sentences": 0, "model_sentences": 0, "praise_examples": [],
             "invites": 0}
    t_start = time.perf_counter()
    rounds = 0
    while True:
        rounds += 1
        for name, turns in conversations:
            s = e.new_session()
            openers, facts_said = [], set()
            stats["conversations"] += 1
            for text, expect in turns:
                seen.clear()
                before = dict(e.fidelity)
                out = list(e.handle_stream(s, text))
                result = out[-1]
                said = [x for x in out if isinstance(x, str)]
                reply = " ".join(said)
                stats["turns"] += 1
                dbg = result.debug
                model_turn = e.last_level == "full" and "allowed" in seen if not a.off else (e.last_level == "full" and dbg.intent not in ("prepare_booking", "confirm_booking", "check_slots", "cancel_draft", "handoff", "end_call"))
                if model_turn:
                    stats["model_replies"] += 1
                    fell = e.fidelity["fell_back"] - before["fell_back"] if not a.off else int("FALLBACK" in (dbg.note or ""))
                    stats["fell_back"] += fell
                    stats["guard_stops"] += e.fidelity["leaks"] - before["leaks"]
                    if fell:
                        why = re.sub(r"\[.*?\]|'[^']*'", "", (dbg.note or "").split("stream")[-1])[:60]
                        stats["fallback_reasons"][why] = stats["fallback_reasons"].get(why, 0) + 1
                    if dbg.lead_ms:
                        stats["first_ms"].append(dbg.lead_ms)
                    stats["ends_on_question"] += reply.rstrip().endswith("?")
                    stats["invites"] += bool(re.search(r"\b(?:test drive|come (?:in|and see|by)|see it in person|visit)\b", reply.lower()))
                    if not fell:
                        for sentence in voice_rules.sentences(reply):
                            stats["model_sentences"] += 1
                            m = PRAISE.search(sentence)
                            if m and m.group(0).lower() not in seen.get("allowed", "").lower():      # a word the facts themselves use ("practical SUV body") is not counted
                                stats["praise_sentences"] += 1
                                if len(stats["praise_examples"]) < 12:
                                    stats["praise_examples"].append(sentence)
                    if "allowed" in seen and not fell:
                        for sentence in voice_rules.sentences(reply):
                            leak = guards.leaks(sentence, seen["allowed"], names)
                            if leak:
                                stats["spoken_leaks"] += 1
                                if len(stats["leak_examples"]) < 10:
                                    stats["leak_examples"].append({"said": sentence, "why": leak})
                    first3 = " ".join(re.findall(r"[a-z']+", reply.lower())[:3])
                    stats["same_opener"] += first3 in openers
                    openers.append(first3)
                for sentence in voice_rules.sentences(reply):
                    key = sentence.strip().lower()
                    if not key.endswith("?") and len(key.split()) >= 5:
                        stats["repeated_fact_sentences"] += key in facts_said
                        facts_said.add(key)
                if expect:
                    stats["tell_me_more"][1] += 1
                    stats["tell_me_more"][0] += expect in reply.lower()
                if a.show:
                    print(f"[{name}] {e.last_level:6s} {dbg.intent:16s} YOU: {text}\n{'':32s}AGENT: {reply}" + (f"\n{'':32s}note: {dbg.note}" if dbg.note else ""), flush=True)
            if name.startswith("mix"):
                stats["bookings"][name] = {"booked": bool(s.booking), "should": int(name[3:]) in BOOKS}
            if a.turns and stats["model_replies"] >= a.turns:
                break
        if not a.turns or stats["model_replies"] >= a.turns:
            break
        print(f"round {rounds}: {stats['model_replies']} model-written replies so far, {time.perf_counter() - t_start:.0f} s", flush=True)
    ms = stats.pop("first_ms")
    stats["first_sentence_ms_median"] = round(statistics.median(ms)) if ms else None
    stats["first_sentence_ms_p90"] = round(sorted(ms)[int(0.9 * (len(ms) - 1))]) if ms else None
    stats["wall_s"] = round(time.perf_counter() - t_start)
    a.out.parent.mkdir(exist_ok=True)
    a.out.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
