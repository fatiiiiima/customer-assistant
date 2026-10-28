"""scripts/replay_turns.py: recorded calls through both model paths against the same backend (Phase 2)."""
import importlib.util
import sys
from pathlib import Path

import pytest

from tests.unit.test_single_call import DATA_JS, WORDS, Fake, engine, turn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def script():
    spec = importlib.util.spec_from_file_location("replay_turns_script", ROOT / "scripts" / "replay_turns.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_callers_turns_are_the_heard_lines_of_a_words_log(tmp_path):
    log = tmp_path / "20261028_000000_0001_in.txt"
    log.write_text("1.0\tsaid\tHi, welcome.\n2.0\tstt\tporsche\n2.1\theard\tPorsche.\n2.2\tsaid\tOkay.\n9.0\tstt\tx\n9.1\theard\t  Test drive.  \n9.2\theard\t\n", encoding="utf-8")
    assert script().read_words_log(log) == ["Porsche.", "Test drive."]


def test_long_digit_runs_are_masked_in_what_is_printed():
    m = script().mask
    assert m("050, 056, 056, 0") == "<number>" and m("my number is 0507932546 thanks") == "my number is <number> thanks"
    assert m("tomorrow at 3") == "tomorrow at 3" and m("a 2016 Macan") == "a 2016 Macan"


def test_the_prefix_share_is_the_part_of_the_second_prompt_that_starts_like_the_first():
    s = script()
    assert s.prefix_share("abcdef", "abcxyz") == 0.5 and s.prefix_share("abc", "abc") == 1.0 and s.prefix_share("x", "") == 0.0


@pytest.mark.skipif(DATA_JS is None, reason="the website is not on this machine")
class TestReplay:
    def run(self, fake, texts, **kw):
        e = engine(fake, single=False)
        rows = script().replay_call(e, texts, log=lambda *_: None, **kw)
        return e, rows

    def test_a_turn_the_code_routes_asks_no_model_and_one_it_cannot_route_is_asked_both_ways(self):
        fake = Fake([turn()], decision="get_car AG-122")
        e, rows = self.run(fake, ["a bmw m3", WORDS])
        assert [r["route"] for r in rows] == ["code", "model"]
        assert fake.calls["decide"] == 1 and fake.calls["structured"] == 1            # once each, and the session's own turn did not ask the model again
        r = rows[1]
        assert r["two"] == ("get_car", "AG-122") and r["one"] == ("get_car", "AG-122") and r["agree"] and r["say"] and r["one_ms"] >= 0 and r["two_ms"] >= 0

    def test_a_different_action_is_a_disagreement_and_is_listed_with_what_each_would_have_meant_to_the_caller(self):
        fake = Fake([turn(action="none", car=None, say="Hello there.")], decision="get_car AG-122")
        _, rows = self.run(fake, ["a bmw m3", WORDS])
        summary = script().summarise(rows)
        d = summary["disagreements"][0]
        assert d["would_say_two_call"] and d["would_say_one_call"] and d["would_say_two_call"] != d["would_say_one_call"] and d["same_words_spoken"] is False
        assert fake.calls["write"] == 0                                                   # (code-level wording: no model was asked to find out)
        assert not rows[1]["agree"] and summary["agree_on_model_turns"] == 0 and summary["agree_share_model_turns"] == 0.0
        assert summary["disagreements"][0]["two_call"] == ("get_car", "AG-122") and summary["disagreements"][0]["one_call"] == ("none", None)
        assert summary["agree_share_all_turns"] == 0.5                                # the code-routed turn agrees by construction

    def test_a_car_named_by_only_one_path_is_the_car_in_focus_not_a_disagreement(self):
        fake = Fake([turn(car="AG-122")], decision="get_car")                             # the two-call decision names no car: the engine uses the one in focus
        _, rows = self.run(fake, ["a bmw m3", WORDS])
        assert rows[1]["agree"] and not rows[1]["agree_strict"]
        assert script().summarise(rows)["agree_strict_on_model_turns"] == 0

    def test_a_malformed_one_call_answer_is_reported_and_is_not_a_disagreement_of_the_action(self):
        fake = Fake(["banana"], decision="get_car AG-122")
        _, rows = self.run(fake, ["a bmw m3", WORDS])
        summary = script().summarise(rows)
        assert summary["malformed"] == 1 and rows[1]["agree"]                           # it fell back to the two-call decision, which is what the engine would do

    def test_a_failed_one_call_is_never_counted_as_agreement(self):
        from dealer_core.nlu.backend import BackendTimeout

        fake = Fake([BackendTimeout("slow")], decision="get_car AG-122")
        _, rows = self.run(fake, ["a bmw m3", WORDS])
        assert not rows[1]["agree"] and rows[1]["one_error"] == "BackendTimeout" and script().summarise(rows)["one_call_errors"] == 1

    def test_the_session_moves_on_with_the_two_call_decision_and_nothing_is_left_patched(self):
        fake = Fake([turn()], decision="get_car AG-122")
        e, rows = self.run(fake, ["a bmw m3", WORDS])
        assert "_decide" not in vars(e) and e.single_call is False and e.governor is None
        assert fake.calls["write"] == 0                                                    # the replay session is at the code level: the model wrote nothing

    def test_the_first_sentence_option_runs_the_whole_turn_both_ways_on_copies(self):
        fake = Fake([turn(), turn()], decision="get_car AG-122")
        e, rows = self.run(fake, ["a bmw m3", WORDS], first_sentence=True)
        r = rows[1]
        assert r["fs_two"]["first_ms"] is not None and r["fs_one"]["first_ms"] is not None
        assert "single call: its reply was used" in r["fs_one"]["note"] and "single call" not in r["fs_two"]["note"]
        assert fake.calls["decide"] == 2 and fake.calls["structured"] == 2                  # each path asked the model for itself, in the measurement and in its whole turn
        assert "_decide" not in vars(e)
        summary = script().summarise(rows)["first_sentence_ms"]
        assert summary["turns"] == 1 and summary["one_call_reply_used"] == 1 and summary["same_spoken_intent"] == 1

    def test_the_order_the_two_paths_are_timed_in_alternates(self):
        order = []
        fake = Fake([turn(), turn()], decision="get_car AG-122")
        orig_structured, orig_lead = fake.structured, fake.lead_in
        fake.structured = lambda *a, **k: (order.append("one"), orig_structured(*a, **k))[1]
        fake.lead_in = lambda *a, **k: (order.append("two") if "TOOL:" in a[1] else None, orig_lead(*a, **k))[1]
        e = engine(fake, single=False)
        script().replay_call(e, ["a bmw m3", WORDS, "my cousin drives a lot, is it comfortable for him"], log=lambda *_: None)
        assert order[:2] == ["two", "one"] and order[2:4] == ["one", "two"]

    def test_the_prompts_cache_share_is_measured_between_consecutive_model_turns(self):
        fake = Fake([turn(), turn()], decision="get_car AG-122")
        _, rows = self.run(fake, ["a bmw m3", WORDS, "my cousin drives a lot, is it comfortable for him"])
        model = [r for r in rows if r["route"] == "model"]
        assert "share_one" not in model[0] and 0 < model[1]["share_one"] <= 1 and 0 < model[1]["share_two"] <= 1
        assert script().summarise(rows)["prefix_share_chars"]["pairs"] == 1


def test_a_summary_of_nothing_does_not_divide_by_zero():
    s = script().summarise([])
    assert s["turns"] == 0 and s["agree_share_model_turns"] is None and s["agree_share_all_turns"] is None
