"""Per-intent precision of the deterministic fast path, with confidence intervals, plus an
out-of-scope false-fire probe. MEASUREMENT ONLY: this script never changes the rules.

Why intervals: a gate of >=98% precision cannot be demonstrated with a handful of examples. If a
rule fired n times with zero errors, the 95% Wilson lower bound is n/(n+z^2) (z=1.96). That reaches
0.98 only at n >= ~188. So the plan is: a rule is *enabled* for an intent only when its measured
lower bound >= 98% on a large labelled set; until then the intent defers to the LLM (and, for
state-changing intents, to a read-back confirmation either way).

Sets: dev (rules were written against it), heldout (I tuned on its misses), fresh (written after
tuning, never tuned on). Only 'fresh' is uncontaminated.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_llm import score  # noqa: E402
from common import save_result  # noqa: E402
from fastpath import classify  # noqa: E402
from nlu_cases import CASES, FRESH, HELDOUT  # noqa: E402

STATE_CHANGING = {
    "booking": ["book_test_drive", "give_slot", "give_contact", "reschedule", "confirm_yes", "confirm_no"],
    "price": ["make_offer", "ask_lowest_price"],
    "cancel": ["cancel_booking"],
    "escalate": ["request_human", "angry", "claim_authority"],
}
HARMLESS = ["greet", "search_cars", "ask_car_info", "ask_policy", "send_details", "goodbye"]
ALL_STATE_CHANGING = {i for v in STATE_CHANGING.values() for i in v}


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    return max(0.0, (centre - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d)


# Out-of-scope probes: written BEFORE looking at any result. Each is placed in a state where the
# trigger intent is VALID (so state-gating cannot save the rules). A rule firing on these is a false fire.
OOS_PROBES: list[tuple[str, str | None, str]] = [
    ("confirm", "S1042", "cancel my Netflix subscription"),
    ("slot", "S1042", "forget about the traffic, it was terrible"),
    ("negotiating", "S1042", "never mind the weather, what time does the football start"),
    ("interest", None, "do you know any good jokes"),
    ("interest", None, "tell me a joke about cars"),
    ("interest", None, "what's the weather like today"),
    ("greeting", None, "can you sing me a song"),
    ("interest", None, "I hate waiting in traffic on Sheikh Zayed Road"),
    ("negotiating", "S1042", "my cousin owns a restaurant in Deira"),
    ("negotiating", "S1042", "my uncle runs the company that makes the tyres"),
    ("answering", "S1042", "I work at the bank across the road, what are your opening hours"),
    ("answering", "S1042", "can you book me a table at a restaurant tonight"),
    ("answering", "S1042", "I want to try out the new restaurant in Marina"),
    ("answering", "S1042", "let's visit Burj Khalifa this weekend"),
    ("answering", "S1042", "send me the pizza menu on whatsapp"),
    ("answering", "S1042", "send me the football match details"),
    ("negotiating", "S1042", "how low can the temperature go in Dubai in winter"),
    ("negotiating", "S1042", "what is the best restaurant you know"),
    ("negotiating", "S1042", "I need to speak to someone about my electricity bill"),
    ("negotiating", "S1042", "the price of petrol is a joke these days"),
    ("interest", None, "my friend is a human rights lawyer"),
    ("confirm", "S1042", "change the channel please"),
    ("confirm", "S1042", "move the sofa to the left"),
    ("slot", "S1042", "today is my birthday"),
    ("slot", "S1042", "I'll have a coffee in the morning"),
    ("contact", "S1042", "my name is Messi and I play football"),
    ("answering", "S1042", "it was a good deal at the supermarket"),
    ("negotiating", "S1042", "we deal with that in the next meeting"),
    ("negotiating", "S1042", "sure, but first tell me about the Dubai Mall opening hours"),
    ("interest", None, "Toyota is a Japanese company with a long history"),
    ("interest", None, "the document is in my bank locker"),
    ("answering", "S1042", "what is the final score of the match"),
]


def main() -> None:
    sets = {"dev": CASES, "heldout": HELDOUT, "fresh": FRESH}
    stats: dict[str, dict[str, dict]] = {}
    for name, cases in sets.items():
        per = defaultdict(lambda: {"fired": 0, "intent_ok": 0, "exact_ok": 0, "wrong": []})
        for state, car, utt, gold in cases:
            r = classify(state, car, utt)
            if r is None:
                continue
            d = per[r.intent]
            d["fired"] += 1
            ok_intent = r.intent == gold["intent"]
            d["intent_ok"] += ok_intent
            d["exact_ok"] += score(gold, json.dumps({"intent": r.intent, "slots": r.slots}))["exact"]
            if not ok_intent:
                d["wrong"].append({"utt": utt, "gold": gold["intent"]})
        stats[name] = dict(per)

    def line(label, d):
        n, k = d["fired"], d["intent_ok"]
        return f"{label:18s} fired {n:3d}  intent-correct {k:3d}  precision {k / n:6.1%}  95% lower bound {wilson_lower(k, n):6.1%}"

    print("== per-intent precision of rule firings (intent level; 'exact' adds slots) ==")
    for group, intents in STATE_CHANGING.items():
        print(f"-- {group}")
        for i in intents:
            for name in ("fresh", "all"):
                if name == "all":
                    agg = {"fired": 0, "intent_ok": 0}
                    for s in sets:
                        d = stats[s].get(i)
                        if d:
                            agg["fired"] += d["fired"]
                            agg["intent_ok"] += d["intent_ok"]
                    d = agg if agg["fired"] else None
                else:
                    d = stats["fresh"].get(i)
                if d:
                    print("   " + line(f"{i} [{name}]", d))
    tot = {"fired": 0, "intent_ok": 0}
    for i in ALL_STATE_CHANGING:
        for s in sets:
            d = stats[s].get(i)
            if d:
                tot["fired"] += d["fired"]
                tot["intent_ok"] += d["intent_ok"]
    print("\n   " + line("ALL state-changing", tot))
    print(f"   examples needed with zero errors for a 98% lower bound: n >= {math.ceil(0.98 * 1.96 ** 2 / 0.02)}")

    print("\n== out-of-scope probes: should NOT fire a state-changing rule ==")
    fired = []
    for state, car, utt in OOS_PROBES:
        r = classify(state, car, utt)
        if r is not None:
            fired.append((state, utt, r.intent))
    unsafe = [f for f in fired if f[2] in ALL_STATE_CHANGING and f[2] != "request_human"]
    handoff = [f for f in fired if f[2] == "request_human"]
    benign = [f for f in fired if f[2] in HARMLESS]
    print(f"probes {len(OOS_PROBES)} | fired anything {len(fired)} | unsafe (state-changing) {len(unsafe)} "
          f"| handoff {len(handoff)} (acceptable) | harmless wrong intent {len(benign)}")
    for state, utt, intent in fired:
        tag = "UNSAFE " if (intent in ALL_STATE_CHANGING and intent != "request_human") else ("handoff" if intent == "request_human" else "benign ")
        print(f"   {tag} [{state}] {utt!r} -> {intent}")
    save_result("precision", {"stats": {k: {i: {kk: vv for kk, vv in d.items()} for i, d in v.items()} for k, v in stats.items()},
                              "oos": {"n": len(OOS_PROBES), "fired": fired, "unsafe": len(unsafe)}})


if __name__ == "__main__":
    main()
