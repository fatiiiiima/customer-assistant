"""The opening line (owner, 30 Oct: option 1 of the greetings offered; docs/ARCH_REVIEW.md section 47): about 3 s instead of 6.6, one question a caller can
answer with a make, a type or a budget; on a car's page, that car and the two things the agent can do with it."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_haggle import _call  # noqa: E402

from dealer_core.config import load_config  # noqa: E402
from dealer_core.dialogue import templates as T  # noqa: E402
from dealer_core.voice.pipeline import common_phrases  # noqa: E402

PLAIN = "Hello, I'm Wazir from AGMC. What kind of car are you after?"


def camry_session():
    F, e, s, said = _call("I'm looking for a Camry.")
    page = e.new_session()
    page.car = s.car
    return F, e, page


def test_the_greeting_is_short_and_asks_one_question():
    F, e, s, said = _call()
    line = e.opening(e.new_session())
    assert line == PLAIN and len(line.split()) == 12 and line.count("?") == 1


def test_on_a_car_s_page_it_names_the_car_and_offers_price_or_test_drive():
    F, e, page = camry_session()
    assert e.opening(page) == "Hello, I'm Wazir from AGMC. I see you're looking at the Camry. Would you like the price, or to book a test drive?"


@pytest.mark.parametrize("answer", ["The price.", "Price.", "The price, please."])
def test_the_price_answer_gives_the_price(answer):
    F, e, page = camry_session()
    e.opening(page)
    intent, reply = F.say(e, page, answer)
    assert intent == "get_car" and "74,500 dirhams" in reply


@pytest.mark.parametrize("answer", ["A test drive.", "Book a test drive."])
def test_the_test_drive_answer_starts_the_booking_for_that_car(answer):
    F, e, page = camry_session()
    e.opening(page)
    intent, reply = F.say(e, page, answer)
    assert intent == "prepare_booking" and "Camry" in reply


@pytest.mark.parametrize("answer, intent", [("A Toyota.", "search_cars"), ("An SUV.", "search_cars"), ("A family car.", "best_of")])
def test_the_answers_the_question_invites_are_understood(answer, intent):
    F, e, s, said = _call()
    e.opening(s)
    assert F.say(e, s, answer)[0] == intent


def test_the_voice_makes_the_greeting_before_the_call():
    """The greeting's first sound is instant only when the line said is word for word the one made at start-up."""
    cfg = load_config(environ={})
    assert common_phrases(cfg)[0] == PLAIN == T.greeting(cfg.app.agent_name)
