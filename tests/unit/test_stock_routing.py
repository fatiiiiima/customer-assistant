"""27 Oct, Chrome: after "There are 4 Toyotas ... Any specific model?" the caller said "I'm not looking for any specific model, what do you have?"
and heard the OTHER MAKES ("Besides Toyota, we've got 20 more cars, from BMW, Audi ...") instead of the Toyota models."""
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from dealer_core.agent.engine import AgentEngine
from dealer_core.app import ROOT, build_app
from dealer_core.config import load_config
from dealer_core.dialogue.states import Intent
from dealer_core.nlu import expected
from dealer_core.nlu.backend import IntentResult

SITE = next((p for p in (ROOT.parent / "bmw-portal" / "site" / "data.js", Path("C:/dev/bmw-portal/site/data.js")) if p.exists()), None)
NOW = datetime(2030, 3, 4, 10, 0, tzinfo=ZoneInfo("Asia/Dubai"))
CALLER = "I'm not looking for any specific model, what do you have?"


class Scripted:
    name = "scripted"

    def classify(self, *a, **k):
        return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

    def lead_in(self, *a, **k):
        return "none"

    def lead_in_stream(self, *a, **k):
        yield "Here you go."


def engine(portal: bool = False):
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml") if portal else load_config(environ={})
    kw = dict(inventory="portal", portal_data=SITE) if portal else dict(inventory="demo")
    app = build_app(cfg, classifier=Scripted(), reply_mode="conversational", clock=lambda: NOW.astimezone(timezone.utc), lead_backend=Scripted(), **kw)
    return AgentEngine(app.service, Scripted(), app.engine.reply_writer)


def turn(e, s, text):
    out = list(e.handle_stream(s, text))
    return out[-1].debug.intent, " ".join(x for x in out if isinstance(x, str))


def toyota_search(e):
    s = e.new_session()
    tool, _ = turn(e, s, "I'm looking for a Toyota")
    assert tool == "search_cars" and s.filters.get("make") == "toyota" and s.shown
    return s


def test_what_do_you_have_during_a_toyota_search_lists_the_toyotas_not_the_other_makes():
    e = engine()
    s = toyota_search(e)
    tool, said = turn(e, s, CALLER)
    assert tool == "search_cars"
    assert "Besides" not in said and "other makes" not in said and "BMW" not in said
    assert s.filters.get("make") == "toyota"                       # the search is still the Toyotas


@pytest.mark.parametrize("words", ["what do you have", "what have you got", "what are my options", "just show me what you have"])
def test_the_other_ways_of_asking_during_a_search_list_the_matches_too(words):
    e = engine()
    s = toyota_search(e)
    assert turn(e, s, words)[0] == "search_cars"


def test_what_do_you_have_with_no_search_under_way_is_still_the_showroom():
    e = engine()
    s = e.new_session()
    tool, said = turn(e, s, "what do you have")
    assert tool == "browse_makes" and "Which make" in said


def test_what_do_you_have_about_the_car_being_discussed_is_still_the_showroom():
    e = engine()
    s = toyota_search(e)
    s.car = s.shown[0].stock_id                                     # the caller picked one: "what do you have" now means beyond it
    assert turn(e, s, "what do you have")[0] == "browse_makes"


def test_the_caller_naming_another_make_is_a_new_search_not_a_relist():
    e = engine()
    s = toyota_search(e)
    tool, said = turn(e, s, "what Honda do you have")
    assert tool != "browse_makes" and s.filters.get("make") != "toyota"


def test_not_looking_for_any_specific_model_is_no_preference():
    assert expected.nothing_in_particular("I'm not looking for any specific model")
    assert expected.nothing_in_particular("I am not looking for anything particular")
    assert not expected.nothing_in_particular("I'm not looking for a Toyota")
    assert not expected.nothing_in_particular("I'm not looking for any specific model but I want a red one")


@pytest.mark.skipif(SITE is None, reason="the website is not on this machine")
def test_the_exact_dialogue_on_the_websites_own_cars():
    e = engine(portal=True)
    s = e.new_session()
    turn(e, s, "I'm looking for a Toyota")
    tool, said = turn(e, s, CALLER)
    assert tool == "search_cars" and "Corolla" in said and "Camry" in said and "RAV4" in said
    assert "BMW" not in said and "other makes" not in said
