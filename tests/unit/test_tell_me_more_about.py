"""ARCH_PLAN Phase 3: "tell me more about X" must answer X. It did, and then added the car's blurb, because "tell me more" was itself read as the topic "about";
"tell me more about its history" got the blurb first and the history second (28 Oct)."""
import sys
from datetime import timezone
from pathlib import Path

import pytest

from dealer_core.agent import answers as A
from dealer_core.agent.engine import AgentEngine
from dealer_core.app import ROOT, build_app
from dealer_core.config import load_config

sys.path.insert(0, str(Path(__file__).parent))
from test_not_repeating import NOW, SITE, Scripted, talk  # noqa: E402


@pytest.mark.parametrize("text, topic", [("tell me more about the engine", "engine"), ("tell me more about the interior", "interior"),
                                         ("tell me more about the fuel economy", "economy"), ("tell me more about the boot", "boot")])
def test_a_named_topic_replaces_the_generic_tell_me_more(text, topic):
    assert A.topics_in(text) == [topic]


def test_tell_me_more_alone_is_still_the_next_group():
    for text in ("tell me more about it", "tell me more", "what else can you tell me", "anything else you can tell me about it"):
        assert A.topics_in(text) == ["about"], text


def test_its_history_is_the_cars_history():
    assert set(A.topics_in("tell me more about its history")) == {"accident", "service"}
    assert A.topics_in("does it have a service history") == ["service"]                 # the existing phrase is unchanged


@pytest.fixture(scope="module")
def site_engine():
    if SITE is None:
        pytest.skip("the website is not on this machine")
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=SITE, classifier=Scripted(), reply_mode="conversational", clock=lambda: NOW.astimezone(timezone.utc), lead_backend=Scripted())
    return AgentEngine(app.service, Scripted(), app.engine.reply_writer)


@pytest.mark.parametrize("asked, said, not_said", [
    ("tell me more about the engine", "horsepower", "sports-car handling"),
    ("tell me more about the interior", "black leather", "sports-car handling"),
    ("tell me more about its history", "service history", "sports-car handling"),
    ("tell me more about the fuel economy", "litres per hundred", "sports-car handling"),
])
def test_the_macan_question_is_answered_and_nothing_else_is_added(site_engine, asked, said, not_said):
    s = site_engine.new_session()
    talk(site_engine, s, "Porsche.")
    talk(site_engine, s, "Yes, talk about the Macan.")
    reply = talk(site_engine, s, asked)[1]
    assert said in reply and not_said not in reply
