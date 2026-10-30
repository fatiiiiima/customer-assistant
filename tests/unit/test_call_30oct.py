"""Found replaying the owner's 30 Oct call (docs/ARCH_REVIEW.md section 44): after a budget search, "Is it white?" about the Camry searched for white
cars and the Camry was lost, so "Can I see it on Tuesday?" then asked which car. The code's own routing (the model too slow or not asked) is what is tested."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_haggle import _call  # noqa: E402


def test_is_it_white_about_the_car_answers_the_colour_and_keeps_the_car():
    F, e, s, said = _call("I'm looking for a Toyota.", "Do you have one under 60,000?", "I'm looking for a Camry.", "Is it white?")
    intent, reply = said[-1]
    camry = s.car
    assert intent == "get_car" and reply.startswith("No, it's brown.") and e.svc.get_car(camry).model == "Camry"


def test_the_right_colour_gets_a_yes():
    F, e, s, said = _call("I'm looking for a Camry.", "Is it brown?")
    assert said[-1][1].startswith("Yes, it's brown.")


def test_then_seeing_it_on_a_day_books_that_car():
    F, e, s, said = _call("I'm looking for a Toyota.", "Do you have one under 60,000?", "I'm looking for a Camry.", "Is it white?", "Can I see it on Tuesday?")
    intent, reply = said[-1]
    assert intent == "prepare_booking" and "Which car" not in reply and "Tuesday" in reply and e.svc.get_car(s.car).model == "Camry"


@pytest.mark.parametrize("text", ["Do you have a white one?", "Have you got anything cheaper?", "Are there any under 100,000?"])
def test_asking_for_other_cars_still_searches(text):
    F, e, s, said = _call("I'm looking for a Camry.", text)
    assert said[-1][0] == "search_cars"


def test_the_colour_question_is_settled_by_the_rules_before_the_model_is_asked():
    """30 Oct test call through the worker: the model, asked, answered "Is it white?" with every colour in stock."""
    F, e, s, said = _call("I'm looking for a Camry.")
    decision = e._fast_decide(s, "Is it white?")
    assert decision is not None and decision["tool"] == "get_car" and decision["car_id"] == s.car


def test_one_car_offered_instead_is_asked_about_as_it_not_which_one():
    """30 Oct: "but I've got the Civic at 54,500 dirhams. Which one would you like to hear more about?" (one car); the model then made it "Tell me more about the Civic?"."""
    F, e, s, said = _call("I'm looking for a Toyota.", "Do you have one under 60,000?")
    assert said[-1][1].endswith("Would you like to hear more about it?") and "Which one" not in said[-1][1]
