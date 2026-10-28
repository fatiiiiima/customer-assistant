"""The single call's prompt is laid out for a server's prefix cache: what never changes first, what changes every turn last (agent/turn_prompt.py, docs/PROMPTS.md)."""
import os.path
from datetime import datetime, timedelta, timezone

import pytest

from dealer_core.agent import turn_prompt
from dealer_core.agent.engine import DECIDE_SYSTEM, TOOLS
from tests.unit.test_single_call import DATA_JS, NOW, WORDS, Fake, at_the_m3, engine

pytestmark = pytest.mark.skipif(DATA_JS is None, reason="the website is not on this machine")


def prompts(e, s, text=WORDS):
    found, ids = e._code_hint(text)
    return e._single_system(), e._single_user(s, text, found, ids)


def common_prefix(a, b):
    return len(os.path.commonprefix([a, b]))


def test_the_sections_are_in_the_order_least_changing_first_and_the_customers_words_last():
    assert turn_prompt.SECTIONS == ("SESSION STATE", "CARS (true facts you may use)", "NOW", "CONVERSATION", "CODE HINT", "CUSTOMER (now)")
    e, s = at_the_m3(Fake())
    _, user = prompts(e, s, "do you have a BMW with leather seats")
    at = [user.index(f"{name}:") for name in turn_prompt.SECTIONS]
    assert at == sorted(at) and len(set(at)) == len(at)
    assert user.rstrip().endswith("CUSTOMER (now): do you have a BMW with leather seats")


def test_a_turn_with_no_match_in_the_stock_leaves_the_hint_out():
    e, s = at_the_m3(Fake())
    _, user = prompts(e, s)
    assert "CODE HINT" not in user


def test_the_system_prompt_names_every_action_the_two_call_decision_does_and_the_dealership_facts():
    e, s = at_the_m3(Fake())
    system, _ = prompts(e, s)
    for tool in TOOLS:
        if tool in DECIDE_SYSTEM:                                           # (browse_makes, fix_booking, which_car and best_of are reached by the code's rules and were never described to the model)
            assert tool in system
    assert DECIDE_SYSTEM.split("\n\n", 1)[1].rstrip("\n") in system         # the same descriptions, word for word
    assert "Wazir" in system and "Sheikh Zayed Road" in system and "09:00 to 20:00" in system and "no discount" in system


def test_the_system_prompt_is_the_same_for_every_caller_every_turn_and_every_minute():
    e1, s1 = at_the_m3(Fake())
    system1, _ = prompts(e1, s1)
    e2, s2 = at_the_m3(Fake())
    e2.svc.clock = lambda: (NOW + timedelta(hours=5, minutes=17)).astimezone(timezone.utc)
    e2.handle(s2, "tell me about the colours")
    assert prompts(e2, s2, "something else entirely")[0] == system1


def test_nothing_that_changes_is_in_the_system_prompt():
    e, s = at_the_m3(Fake())
    system, _ = prompts(e, s)
    assert "AG-122" not in system and "M3" not in system and "2030" not in system and "NOW" not in system and "March" not in system


def test_two_turns_in_a_row_share_everything_up_to_the_conversation_and_the_clock():
    e, s = at_the_m3(Fake())
    system, u1 = prompts(e, s, "my cousin drives a lot, is it comfortable")
    e.handle(s, "my cousin drives a lot, is it comfortable")                  # one more exchange
    _, u2 = prompts(e, s, "what would my wife think of it")
    shared = common_prefix(system + u1, system + u2)
    assert shared >= len(system) + u1.index("CONVERSATION:")                  # the instructions, the state and the cars are one cached block
    assert shared < len(system) + len(u1)                                     # (and the turns differ)


def test_the_clock_changes_only_the_part_after_the_cars():
    e, s = at_the_m3(Fake())
    system, u1 = prompts(e, s)
    e.svc.clock = lambda: (NOW + timedelta(minutes=1)).astimezone(timezone.utc)
    _, u2 = prompts(e, s)
    assert common_prefix(u1, u2) == u1.index("NOW:") + len("NOW: Monday 04 March 2030, 10:0")


def test_the_cars_in_play_carry_their_stock_id_for_the_model_to_choose_from():
    e, s = at_the_m3(Fake())
    _, user = prompts(e, s)
    assert "- [AG-122] 2021 BMW M3: green" in user and "CAR IN FOCUS: AG-122" in user


def test_the_sliding_window_is_the_writers_history_length():
    e, s = at_the_m3(Fake())
    for i in range(6):
        s.remember("customer", f"line {i}")
        s.remember("agent", f"reply {i}")
    _, user = prompts(e, s)
    shown = [l for l in user.splitlines() if l.startswith(("AGENT:", "CUSTOMER:")) and not l.startswith("CUSTOMER (now)")]
    assert len(shown) == e.reply_writer.history_lines
