"""Lines from real calls, decided by code (via: rules). The ones that may need the model run with scripts/replay_utterances.py."""
from pathlib import Path

import pytest
import yaml

from dealer_core.dialogue.states import Intent, State
from dealer_core.nlu.rules import RuleEngine, load_enabled_intents

CASES = yaml.safe_load((Path(__file__).resolve().parents[1] / "data" / "real_utterances.yaml").read_text(encoding="utf-8"))
RULE_CASES = [c for c in CASES if c["via"] == "rules"]
VOCAB = {"toyota": ["corolla", "camry", "rav4", "land cruiser"], "bmw": ["m3"], "kia": ["carnival"]}


@pytest.mark.parametrize("case", RULE_CASES, ids=[c["name"] for c in RULE_CASES])
def test_rules_decide_it(case):
    rules = RuleEngine(lambda: VOCAB, load_enabled_intents(Path(__file__).resolve().parents[2] / "config" / "nlu_gates.yaml"))
    hit = rules.classify(State(case["state"]), case.get("car"), case["text"], case["last_agent"])
    assert hit is not None and hit.intent is Intent(case["intent"]), f"{case['text']!r} after {case['last_agent']!r}: {hit}"


def test_a_few_unclear_words_get_the_question_asked_again_not_a_colleague():
    from scenarios.conftest import Call

    c = Call({"hmm banana": "ask_policy"})
    c.s.state = State.INTEREST
    c.app.engine.opening(c.s)
    r = c.say("hmm banana")
    assert r.reply.startswith("Sorry, I didn't quite catch that.") and "colleague" not in r.reply and r.reply.endswith("?")
