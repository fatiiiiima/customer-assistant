"""The single-call turn (`dialogue.single_call`, Phase 2): one request returns the action and the reply for a turn the code cannot route; anything wrong falls back to the two-call path."""
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from dealer_core.agent import turn_schema
from dealer_core.agent.engine import DECIDE_SYSTEM, TOOLS, AgentEngine
from dealer_core.agent.governor import QualityGovernor
from dealer_core.app import ROOT, build_app
from dealer_core.config import load_config
from dealer_core.dialogue.states import Intent
from dealer_core.nlu.backend import BackendTimeout, IntentResult

DATA_JS = next((p for p in (Path("C:/dev/bmw-portal/site/data.js"), Path(__file__).resolve().parents[3] / "bmw-portal" / "site" / "data.js") if p.exists()), None)
pytestmark = pytest.mark.skipif(DATA_JS is None, reason="the website is not on this machine")
NOW = datetime(2030, 3, 4, 10, 0, tzinfo=ZoneInfo("Asia/Dubai"))
WORDS = "my cousin drives a lot, is it comfortable"          # the code does not route this (checked below): the model decides
GOOD_SAY = "It seats 5 and has 510 hp."


def turn(action="get_car", car="AG-122", say=GOOD_SAY):
    return json.dumps({"action": action, "args": {"car_id": car}, "say": say})


class Fake:
    """The model: answers the single call from a list (a string, or an exception to raise), counts every kind of call."""
    name = "fake"

    def __init__(self, answers=(), decision="none", has_structured=True):
        self.answers, self.decision, self.calls, self.seen = list(answers), decision, {"structured": 0, "decide": 0, "write": 0}, []
        if not has_structured:
            self.structured = None

    def classify(self, *a, **k):
        return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

    def structured(self, system, user, schema, max_tokens=120, deadline_s=None):
        self.calls["structured"] += 1
        self.seen.append((system, user, schema, max_tokens, deadline_s))
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    def lead_in(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
        if "TOOL:" in user:
            self.calls["decide"] += 1
            return self.decision
        return "ok."

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        self.calls["write"] += 1
        yield "It seats 5, so it suits a family."


def engine(fake, single=True, governor=None):
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=DATA_JS, classifier=fake, reply_mode="conversational",
                    clock=lambda: NOW.astimezone(timezone.utc), lead_backend=fake)
    e = AgentEngine(app.service, fake, app.engine.reply_writer)
    e.single_call, e.governor = single, governor
    return e


def at_the_m3(fake, **kw):
    e = engine(fake, **kw)
    s = e.new_session()
    e.handle(s, "a bmw m3")                                              # routed by code, no model call
    assert s.car == "AG-122" and fake.calls["structured"] == fake.calls["decide"] == 0      # the code routed it; (the writer was asked for the words)
    fake.calls.update(structured=0, decide=0, write=0)
    return e, s


def test_the_words_used_here_are_ones_the_code_does_not_route():
    fake = Fake()
    e, s = at_the_m3(fake)
    assert e._fast_decide(s, WORDS) is None


def test_one_request_gives_the_action_and_the_reply_and_nothing_else_is_asked():
    fake = Fake([turn()])
    e, s = at_the_m3(fake)
    r = e.handle(s, WORDS)
    assert fake.calls == {"structured": 1, "decide": 0, "write": 0}
    assert GOOD_SAY in r.reply and r.debug.intent == "get_car"
    assert "single call: its reply was used" in r.debug.note and s.car == "AG-122"


def test_the_replys_facts_come_from_the_tool_not_from_the_model():
    fake = Fake([turn(say="It seats 7 and has 900 hp.")])               # numbers that are not the car's
    e, s = at_the_m3(fake)
    r = e.handle(s, WORDS)
    assert "900" not in r.reply and "single call: its reply was refused" in r.debug.note
    assert fake.calls["write"] == 1                                       # the writer was asked, as on the two-call path


def test_a_malformed_answer_is_treated_as_no_answer_and_the_two_call_path_decides():
    fake = Fake(['{"action": "get_car", "args": ', "banana"], decision="get_car AG-122")
    e, s = at_the_m3(fake)
    r = e.handle(s, WORDS)
    assert fake.calls["structured"] == 1 and fake.calls["decide"] == 1 and fake.calls["write"] == 1
    assert r.debug.intent == "get_car" and "answer malformed" in r.debug.note and "two-call decision was asked" in r.debug.note


@pytest.mark.parametrize("bad", [turn(action="launch_rocket"), turn(car="ZZ-999"), json.dumps(["get_car"]), turn(say="x" * 400)])
def test_each_kind_of_invalid_answer_falls_back(bad):
    fake = Fake([bad], decision="get_car AG-122")
    e, s = at_the_m3(fake)
    r = e.handle(s, WORDS)
    assert fake.calls["decide"] == 1 and r.debug.intent == "get_car"


def test_a_timeout_is_handled_as_a_timeout_of_the_decision_the_code_decides_and_nobody_asks_twice():
    fake = Fake([BackendTimeout("too slow")], decision="search_cars")
    e, s = at_the_m3(fake)
    r = e.handle(s, WORDS)
    assert fake.calls["decide"] == 0 and fake.calls["structured"] == 1
    assert "model too slow: decided by code" in r.debug.note and r.debug.intent == "get_car"      # the code's own decision: the car being discussed


def test_the_one_call_has_the_first_sentence_deadline():
    fake = Fake([turn()])
    e, s = at_the_m3(fake)
    e.single_deadline_s = 1.5
    e.handle(s, WORDS)
    assert fake.seen[0][4] == 1.5


def test_a_backend_that_cannot_hold_a_schema_uses_the_two_call_path_without_fuss():
    fake = Fake(has_structured=False, decision="get_car AG-122")
    e, s = at_the_m3(fake)
    r = e.handle(s, WORDS)
    assert fake.calls["decide"] == 1 and fake.calls["write"] == 1 and "single call" not in r.debug.note


def test_a_server_that_refuses_the_schema_gives_none_and_the_two_call_path_runs():
    fake = Fake([None], decision="get_car AG-122")                         # (the OpenAI-compatible backend returns None on HTTP 400)
    e, s = at_the_m3(fake)
    e.handle(s, WORDS)
    assert fake.calls["decide"] == 1


def test_off_by_default_so_the_two_call_path_is_what_runs():
    assert load_config(environ={}).dialogue.single_call is False
    fake = Fake(decision="get_car AG-122")
    e, s = at_the_m3(fake, single=False)
    e.handle(s, WORDS)
    assert fake.calls == {"structured": 0, "decide": 1, "write": 1}


def test_codes_routing_comes_first_the_model_is_not_asked_for_what_the_code_knows():
    fake = Fake()
    e, s = at_the_m3(fake)
    e.handle(s, "how many kilometres has it done")
    assert fake.calls["structured"] == 0 and fake.calls["decide"] == 0


def test_the_other_actions_keep_their_own_replies_a_say_is_ignored():
    fake = Fake([turn(action="showroom_info", car=None, say="We have lots of lovely cars.")])
    e, s = at_the_m3(fake)
    r = e.handle(s, WORDS)
    assert "lovely cars" not in r.reply and fake.calls["structured"] == 1 and r.debug.intent == "showroom_info"


def test_booking_lines_stay_exact_code_whatever_the_model_wrote():
    fake = Fake([turn(action="prepare_booking", say="Your test drive is booked and confirmed for tomorrow at nine.")])
    e, s = at_the_m3(fake)
    r = e.handle(s, WORDS)
    assert "booked" not in r.reply.lower() and "confirmed" not in r.reply.lower() and not s.booking


def test_a_reply_written_for_an_action_the_code_then_changes_is_not_used():
    fake = Fake([turn(action="none", car=None, say="Lovely, I will note that down for you.")])
    e, s = at_the_m3(fake)
    e.handle(s, "can I book a test drive")                                  # (a booking is under way: the code reads a pick from the list in the next words even if the model says "none")
    fake.calls.update(structured=0, decide=0, write=0)
    r = e.handle(s, "my cousin says the first one is nicer")
    assert fake.calls["structured"] == 1 and r.debug.intent == "get_car"
    assert "note that down" not in r.reply and "the code changed the action or the car" in r.debug.note
    assert fake.calls["write"] == 1                                         # the writer is asked for the action the code really ran


def test_the_model_is_asked_for_a_reply_only_when_the_governor_will_have_it_written():
    for level, expect in (("full", 1), ("prefix", 0), ("code", 0)):
        fake = Fake([turn()] if expect else [], decision="get_car AG-122")
        e, s = at_the_m3(fake, governor=QualityGovernor(level))
        e.handle(s, WORDS)
        assert fake.calls["structured"] == expect, level
    fake = Fake([turn()], decision="get_car AG-122")                         # auto: the measured first-sentence time against the budget
    gov = QualityGovernor("auto", budget_ms=700)
    gov.stats.full_first_ms = 1400
    e, s = at_the_m3(fake, governor=gov)
    e.handle(s, WORDS)
    assert fake.calls["structured"] == 0
    gov.stats.full_first_ms = 400
    e.handle(s, WORDS)
    assert fake.calls["structured"] == 1


def test_the_governor_is_told_the_time_of_the_decision_step_and_not_that_the_reply_cost_nothing():
    fake = Fake([turn()])
    gov = QualityGovernor("full")
    e, s = at_the_m3(fake, governor=gov)
    gov.stats.full_first_ms = gov.stats.decide_ms = None            # (the first, routed turn timed its own writer call)
    e.handle(s, WORDS)
    assert gov.stats.decide_ms is not None            # the step the caller waited for
    assert gov.stats.full_first_ms is None            # the reply's 0 ms was not a measurement of the model: it would make a slow model look fast


def test_a_refused_reply_does_not_make_the_writers_time_count_either_way_it_is_the_two_call_path():
    fake = Fake([turn(say="It seats 7 and has 900 hp.")])
    gov = QualityGovernor("full")
    e, s = at_the_m3(fake, governor=gov)
    gov.stats.full_first_ms = None
    e.handle(s, WORDS)
    assert gov.stats.full_first_ms is not None        # the writer was asked: its first sentence is timed as always
