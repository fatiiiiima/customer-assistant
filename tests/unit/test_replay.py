from dealer_core.cli.replay import check, run_once
from dealer_core.dialogue.machine import TurnDebug


def dbg(**kw):
    base = dict(state_before="a", state_after="b", intent="search_cars", via="rules:filters", tools=["search_cars(x) -> 3 match(es)"],
                decision="counter amount=147100 round=1 rounds_left=2")
    base.update(kw)
    return TurnDebug(**base)


def test_check_passes_when_everything_matches():
    exp = {"intent": "search_cars", "via": "rules", "state_after": "b", "ended": False, "reply_has": ["SUVs"], "reply_not": ["floor"],
           "tool_has": ["search_cars"], "decision_has": ["counter"]}
    assert check(exp, "I have 6 SUVs", dbg(), False) == []


def test_check_reports_each_kind_of_mismatch():
    exp = {"intent": "make_offer", "via": "llm", "state_after": "c", "ended": True, "reply_has": ["147,100"], "reply_not": ["SUVs"],
           "tool_has": ["propose_price"], "decision_has": ["escalate"]}
    bad = check(exp, "I have 6 SUVs", dbg(), False)
    joined = " | ".join(bad)
    for part in ("intent", "handled by", "state_after", "ended", "lacks '147,100'", "contains 'SUVs'", "propose_price", "escalate"):
        assert part in joined, part


def test_check_flags_voice_rule_violations_even_without_expectations():
    assert any("voice-rule" in b for b in check({}, "x", dbg(violations=["3 sentences (max 2)"]), False))


def test_run_once_stops_and_reports_when_the_call_ends_early():
    from scenarios.conftest import Call

    c = Call({"you are useless": "angry"})
    scenario = {"steps": [{"say": "you are useless", "expect": {"intent": "angry", "ended": True}},
                          {"say": "hello again", "expect": {}}]}
    failures, llm_ms = run_once(c.app, scenario, show=False, debug=False, pause=0)
    assert any("ended before the script did" in f for f in failures) and len(llm_ms) == 1     # the one turn the (fake) LLM handled
