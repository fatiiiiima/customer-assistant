"""Too many cars match: the one question asked is about what the caller has not said yet (owner, 30 Oct: "Something under 100,000." found 8 cars, then
"Could you tell me your budget or a make?"; docs/ARCH_REVIEW.md section 48)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_haggle import _call  # noqa: E402

from dealer_core.agent.wording import NARROW_QUESTIONS, narrow_question  # noqa: E402
from dealer_core.config import load_config  # noqa: E402
from dealer_core.voice.pipeline import common_phrases  # noqa: E402


@pytest.mark.parametrize("said, asked", [
    ("Something under 100,000.", "Is there a make you like, or a type of car, like an SUV?"),
    ("An SUV under 150,000.", "Is there a make you like?"),
    ("An SUV.", "Could you tell me your budget or a make?"),
    ("A Toyota.", "What budget do you have in mind, or is there a model you like?"),
])
def test_the_question_is_about_what_was_not_said(said, asked):
    F, e, s, turns = _call(said)
    intent, reply = turns[-1]
    assert intent == "search_cars" and reply.endswith(asked)


@pytest.mark.parametrize("filters", [{"max_price": 100000}, {"min_price": 50000}, {"max_price": 100000, "body_type": "suv"}, {"make": "bmw", "max_price": 200000},
                                     {"make": "bmw", "max_price": 200000, "body_type": "suv"}])
def test_a_budget_given_is_never_asked_for_again(filters):
    assert "budget" not in narrow_question(filters)


def test_a_make_given_is_never_asked_for_again():
    assert all("make" not in narrow_question(f) for f in ({"make": "toyota"}, {"make": "toyota", "max_price": 100000}, {"make": "toyota", "body_type": "suv"}))


def test_the_model_is_told_only_what_is_missing():
    """At the prefix level the model rewords the question: it is told which things are still unknown, not "budget, make, body type or colour"."""
    F, e, s, turns = _call()
    seen, search = [], e._t_search_cars
    e._t_search_cars = lambda *a: seen.append(search(*a)) or seen[-1]          # what the search tool gives the model, on a real turn
    F.say(e, e.new_session(), "Something under 100,000.")
    out = seen[-1]
    line = next(x for x in out["facts"] if x.startswith("Ask one narrowing question"))
    assert line.endswith("not said yet: make, body type, colour.")


def test_every_question_is_made_by_the_voice_before_the_call():
    phrases = common_phrases(load_config(environ={}))
    assert all(q in phrases for q in NARROW_QUESTIONS)
