"""28 Oct, Chrome, the Macan call. The agent asked "Would you like to come and see it?" and the caller said "Yes, I want to come and see the Macan." The agent answered with
more facts (seats, boot size) instead of starting the booking, three times. The words "the Macan" matched a pick from the list just shown, and that rule ran before the
rule for "come and see"."""
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from dealer_core.agent.engine import AgentEngine
from dealer_core.app import ROOT, build_app
from dealer_core.config import load_config
from dealer_core.dialogue.states import Intent, State
from dealer_core.nlu.backend import IntentResult

SITE = next((p for p in (ROOT.parent / "bmw-portal" / "site" / "data.js", Path("C:/dev/bmw-portal/site/data.js")) if p.exists()), None)
NOW = datetime(2026, 10, 28, 11, 0, tzinfo=ZoneInfo("Asia/Dubai"))
pytestmark = pytest.mark.skipif(SITE is None, reason="needs the website's data.js (the Macan)")


class Model:
    """The model, if it is asked: it always says get_car, the answer the live call showed."""
    name = "scripted"

    def classify(self, *a, **k):
        return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

    def lead_in(self, system, user, *a, **k):
        return "get_car" if "TOOL:" in user else "ok"

    def lead_in_stream(self, *a, **k):
        yield "Here you go."


def engine():
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=SITE, classifier=Model(), reply_mode="conversational", clock=lambda: NOW.astimezone(timezone.utc), lead_backend=Model())
    return AgentEngine(app.service, Model(), app.engine.reply_writer)


def turn(e, s, text):
    out = list(e.handle_stream(s, text))
    return out[-1].debug.intent, " ".join(x for x in out if isinstance(x, str))


def macan_call(e):
    s = e.new_session()
    turn(e, s, "Looking for a Porsche.")
    turn(e, s, "Tell me about the Macan.")
    turn(e, s, "I want to hear more about the Macan.")
    s.state = State.ANSWERING
    return s


@pytest.mark.parametrize("text", ["Yes, I want to come and see the Macan.", "I want to come and see the Macan.", "Can I come and see the Macan?",
                                  "I want to visit and see the Macan", "yes, I'd like to come and see it", "Can I test drive the Macan?", "book a test drive for the Macan"])
def test_wanting_to_come_and_see_the_car_starts_the_booking(text):
    e = engine()
    s = macan_call(e)
    intent, said = turn(e, s, text)
    assert intent != "get_car", said
    assert "kilometres" not in said and "boot" not in said and "seats" not in said
    assert "day" in said.lower() or "time" in said.lower() or "when" in said.lower() or "name" in said.lower()      # it asks when


def test_it_is_the_same_before_any_car_is_in_focus_but_one_is_named():
    e = engine()
    s = e.new_session()
    turn(e, s, "Looking for a Porsche.")
    intent, said = turn(e, s, "I want to come and see the Macan.")
    assert intent != "get_car", said


@pytest.mark.parametrize("text", ["I want to hear more about the Macan.", "Tell me about the Macan.", "Yes, tell me about the Macan."])
def test_asking_about_the_car_still_gives_facts(text):
    e = engine()
    s = macan_call(e)
    intent, _ = turn(e, s, text)
    assert intent == "get_car"
