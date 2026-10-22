"""The agent engine with a scripted 'model': it names a tool, code fills the arguments from the caller's words and runs the tool for real."""
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from dealer_core.agent.engine import AgentEngine, _json_from
from dealer_core.app import ROOT, build_app
from dealer_core.config import load_config
from dealer_core.nlu.backend import IntentResult
from dealer_core.dialogue.states import Intent

DATA_JS = Path("C:/dev/bmw-portal/site/data.js")
NOW = datetime(2030, 3, 4, 10, 0, tzinfo=ZoneInfo("Asia/Dubai"))


class Scripted:
    """Stands in for the model: decisions come from a list, replies are a fixed friendly sentence."""
    name = "scripted"

    def __init__(self, decisions):
        self.decisions = list(decisions)

    def classify(self, *a, **k):
        return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

    def lead_in(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
        return self.decisions.pop(0) if "TOOL:" in user else "ok"

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
        yield "Here you go. Anything else?"


def engine(decisions):
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=DATA_JS, classifier=Scripted([]), reply_mode="conversational",
                    clock=lambda: NOW.astimezone(timezone.utc), lead_backend=Scripted(decisions))
    return AgentEngine(app.service, Scripted(decisions), app.engine.reply_writer), app


def test_json_is_found_inside_chatter():
    assert _json_from('Sure: {"tool": "get_car", "args": {"car_id": "AG-101"}} done') == {"tool": "get_car", "args": {"car_id": "AG-101"}}
    assert _json_from("no json here") == {}


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_a_booking_needs_a_read_back_and_a_yes_and_is_then_made_by_code():
    e, app = engine(["search_cars", "prepare_booking AG-122", "prepare_booking AG-122", "prepare_booking AG-122", "confirm_booking"])
    s = e.new_session()
    e.handle(s, "a bmw m3")
    assert s.car == "AG-122"
    e.handle(s, "tomorrow at four in the afternoon")
    assert s.draft_slot is not None and not s.booking
    e.handle(s, "my name is Sara Khan")
    r = e.handle(s, "my number is zero five zero one two three four five six seven")
    assert s.read_back and not s.booking                                  # read back, not yet booked
    e.handle(s, "yes")
    assert s.booking and s.booking.customer_name == "Sara Khan" and s.booking.stock_id == "AG-122"


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_confirm_without_a_read_back_books_nothing():
    e, app = engine(["confirm_booking"])
    s = e.new_session()
    e.handle(s, "yes")
    assert not s.booking


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_an_unknown_tool_from_the_model_is_treated_as_no_tool():
    e, app = engine(["delete_everything"])
    s = e.new_session()
    assert e.handle(s, "hello").debug.intent == "none"
