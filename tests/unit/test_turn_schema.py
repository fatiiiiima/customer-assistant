"""The answer of the single-call turn: one schema, one check (agent/turn_schema.py)."""
import json

import pytest

from dealer_core.agent.engine import TOOLS
from dealer_core.agent.turn_schema import MAX_SAY_CHARS, SAY_ACTIONS, MalformedTurn, Turn, parse_turn, same_action, turn_schema

IDS = ["AG-101", "AG-122"]


def ok(**kw):
    d = {"action": "get_car", "args": {"car_id": "AG-101"}, "say": "It has 91,000 kilometres."}
    d.update(kw)
    return json.dumps(d)


def test_the_schema_lists_every_tool_and_only_the_cars_in_play_and_the_reply_comes_last():
    sch = turn_schema(TOOLS, IDS)
    assert sch["properties"]["action"]["enum"] == list(TOOLS)
    assert sch["properties"]["args"]["properties"]["car_id"]["enum"] == [*IDS, None]
    assert list(sch["properties"]) == ["action", "args", "say"] and sch["required"] == ["action", "args", "say"]
    assert sch["properties"]["say"]["maxLength"] == MAX_SAY_CHARS and sch["additionalProperties"] is False


def test_the_schema_is_plain_json():
    json.dumps(turn_schema(TOOLS, []))


def test_a_valid_answer_is_read():
    assert parse_turn(ok(), TOOLS, IDS) == Turn("get_car", "AG-101", "It has 91,000 kilometres.")


def test_a_think_block_and_extra_spaces_are_tolerated():
    t = parse_turn("<think>hm</think>" + ok(say="  It   has 91,000 kilometres. "), TOOLS, IDS)
    assert t.say == "It has 91,000 kilometres."


def test_no_car_is_fine_for_an_action_that_needs_none():
    assert parse_turn(ok(action="none", args={"car_id": None}, say="Hello!"), TOOLS, IDS).car_id is None
    assert parse_turn(ok(action="none", args={}, say="Hello!"), TOOLS, IDS).car_id is None
    assert parse_turn(ok(action="none", args=None, say="Hello!"), TOOLS, IDS).car_id is None


def test_a_reply_with_no_full_stop_gets_one():
    assert parse_turn(ok(say="It has 91,000 kilometres"), TOOLS, IDS).say.endswith(".")
    assert parse_turn(ok(say="Which one?"), TOOLS, IDS).say == "Which one?"


def test_a_reply_is_kept_only_for_the_actions_that_have_one():
    assert set(SAY_ACTIONS) == {"get_car", "none"}
    for action in TOOLS:
        t = parse_turn(ok(action=action, say="Whatever."), TOOLS, IDS)
        assert (t.say == "Whatever.") is (action in SAY_ACTIONS), action


@pytest.mark.parametrize("raw, why", [
    ("", "not JSON"), ("banana", "not JSON"), ('{"action": "get_car", "args": ', "not JSON"),
    ("[1, 2]", "not a JSON object"), ('"get_car"', "not a JSON object"),
    (ok(action="launch_rocket"), "unknown action"), (json.dumps({"args": {}, "say": ""}), "unknown action"),
    (ok(args={"car_id": "ZZ-9"}), "not one of the cars in play"),
    (ok(args=["AG-101"]), "args is not an object"), (ok(say=7), "say is not text"),
    (ok(say="x" * (MAX_SAY_CHARS + 1)), "characters"),
])
def test_each_kind_of_invalid_answer_is_refused_with_its_reason(raw, why):
    with pytest.raises(MalformedTurn, match=why):
        parse_turn(raw, TOOLS, IDS)


def test_same_action_looks_at_the_car_only_where_the_action_is_about_a_car():
    assert same_action(("get_car", "AG-101"), ("get_car", "AG-101"))
    assert not same_action(("get_car", "AG-101"), ("get_car", "AG-122"))
    assert not same_action(("get_car", "AG-101"), ("search_cars", None))
    assert same_action(("search_cars", None), ("search_cars", "AG-101"))          # a car on an action that is not about one means nothing
    assert same_action(("get_car", ""), ("get_car", None))
