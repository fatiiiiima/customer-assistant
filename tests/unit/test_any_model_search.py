"""29 Oct, the owner's call: after "There are 4 Toyotas... What budget do you have in mind, or is there a model you like?", "Any model and any budget is all you need."
and "Any model or any budget is alright." were answered "a make, a type of car, or a budget?". The search was still Toyota; the phrase was not recognised, and the code's own
decision (the model too slow) ignored the search under way. docs/ARCH_REVIEW.md section 26."""
import sys
from pathlib import Path

import pytest

from dealer_core.nlu import expected

sys.path.insert(0, str(Path(__file__).parent))
from test_test_call_28oct import Scripted, TooSlow, engine, say  # noqa: E402

LINES = ["I'm looking for a Toyota.", "Any model and any budget is all you need.", "Any model or any budget is alright.", "Any model, Toyota."]


@pytest.mark.parametrize("text", ["any model and any budget is all you need", "any model or any budget is alright", "any budget is okay", "any model please",
                                  "no preference either way", "any make is fine with me", "anything at all", "any budget", "any model and any budget"])
def test_no_preference_with_an_ordinary_ending_is_understood(text):
    assert expected.nothing_in_particular(text)


@pytest.mark.parametrize("text", ["any toyota under 80000", "i want a red one", "is it any good", "anything cheaper", "any updates on my booking"])
def test_but_not_a_request_with_something_in_it(text):
    assert not expected.nothing_in_particular(text)


@pytest.mark.parametrize("decisions", [TooSlow(), Scripted()], ids=["model too slow", "model says none"])
def test_the_toyota_call_reads_out_the_toyotas_instead_of_asking_for_a_make_again(decisions):
    e = engine(TooSlow(), decisions=decisions)
    s = e.new_session()
    replies = [say(e, s, t) for t in LINES]
    for intent, reply in replies[1:3]:
        text = " ".join(reply)
        assert intent == "search_cars" and "Corolla" in text and "make, a type of car" not in text and "particular make" not in text
    assert s.filters.get("make") == "toyota"                                     # the search was remembered all along


def test_the_code_alone_reads_out_the_search_under_way_rather_than_asking_again():
    e = engine(TooSlow())
    s = e.new_session()
    say(e, s, "I'm looking for a Toyota.")
    assert e._code_decision(s, "hmm, whatever you think")["tool"] == "search_cars"
    fresh = e.new_session()
    assert e._code_decision(fresh, "hmm, whatever you think").get("fallback")    # nothing under way: still asks what they are after
