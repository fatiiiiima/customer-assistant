"""27 Oct, the Macan call: in prefix mode the model rewords the closing question the code asks, and five of nine times it asked something else:
"Shall we talk about the next available model?" for "hear more, or book a test drive?", "How about its driving experience or features?" for "anything else about it?".
Now a closing question may only be the code's question in other words; anything else falls back to the code's own question."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from dealer_core.agent.engine import AgentEngine, TurnDebug
from dealer_core.app import build_app
from dealer_core.config import load_config
from dealer_core.dialogue.states import Intent
from dealer_core.nlu.backend import IntentResult

NOW = datetime(2026, 10, 27, 11, 0, tzinfo=ZoneInfo("Asia/Dubai"))
FACTS = "Prices are fixed (no negotiation).\n2016 Porsche Macan S, black, 91,000 km, Petrol, 340 hp"
PREFIX = "It's the 2016 Porsche Macan in black, with 91,000 kilometres on it, at 149,000 dirhams."


class Backend:
    """The model: the sentences it will write, one per attempt."""
    name = "scripted"

    def __init__(self, sentences=()):
        self.sentences, self.calls = list(sentences), 0

    def classify(self, *a, **k):
        return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

    def lead_in(self, *a, **k):
        return "none"

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        self.calls += 1
        yield self.sentences.pop(0) if self.sentences else "Anything else?"


def engine(sentences=()):
    app = build_app(load_config(environ={}), inventory="demo", classifier=Backend(), reply_mode="conversational", clock=lambda: NOW.astimezone(timezone.utc),
                    lead_backend=Backend())
    e = AgentEngine(app.service, Backend(), app.engine.reply_writer)
    e.reply_writer.backend = Backend(sentences)
    return e


def problems(e, sentence, question, prefix=PREFIX, caller="tell me more"):
    return e._check_continuation(sentence, prefix, question, caller, FACTS, "")


# ------------------------------------------------------------------------------------------------ what drifted, word for word from the live replay
@pytest.mark.parametrize("sentence,question", [
    ("Shall we talk about the next available model?", "Would you like to hear more, or book a test drive?"),
    ("How about its driving experience or features?", "What else would you like to know about it?"),
    ("How about its daily drive feel or comfort features?", "Anything else you'd like to know about it?"),
    ("Does that sound like the kind of car you'd enjoy driving?", "What else would you like to know about it?"),
    ("Would you like to know more about its driving dynamics?", "Anything else you'd like to know about it?"),
    ("Maybe you'd like to take a drive and feel it for yourself?", "Anything else you'd like to know about it?"),
    ("How about its interior comfort for everyday use?", "What else would you like to know about it?"),
])
def test_a_question_about_something_the_code_did_not_ask_is_refused(sentence, question):
    p = problems(engine(), sentence, question)
    assert any("did not ask about" in x for x in p), p


@pytest.mark.parametrize("sentence,question", [
    ("Would you fancy coming in to see it?", "Would you like to come and see it?"),
    ("Is there anything else you'd like to hear about it?", "Anything else you'd like to know about it?"),
    ("Shall I set up a test drive for you?", "Want to book a test drive?"),
    ("Would you like to hear more about it, or shall we book a test drive?", "Would you like to hear more, or book a test drive?"),
    ("Anything else you'd like to know?", "What else would you like to know about it?"),
    ("Which one would you like to hear more about?", "Which one would you like to hear more about?"),
])
def test_the_same_question_in_other_words_is_allowed(sentence, question):
    assert problems(engine(), sentence, question) == []


def test_a_word_from_the_facts_or_the_callers_own_words_is_not_new():
    e = engine()
    assert problems(e, "Would you like to see the Macan in person?", "Would you like to come and see it?") == []              # Macan is in what was said
    assert problems(e, "Anything else about the horsepower?", "Anything else you'd like to know about it?", caller="what about the horsepower") == []
    assert any("did not ask about" in x for x in problems(e, "Anything else about the horsepower?", "Anything else you'd like to know about it?", prefix="It is black."))


def test_endings_do_not_make_a_word_new():
    assert engine()._novel_words("Would you like me to arrange viewing times?", "Would you like to arrange a viewing time?") == set()


# ------------------------------------------------------------------------------------------------ what is finally said
def spoken(sentences):
    e = engine(sentences)
    s = e.new_session()
    out = list(e._prefix_reply(s, "tell me more", {"facts": FACTS.split("\n")}, PREFIX, "What else would you like to know about it?", TurnDebug(state_before="answering")))
    return e, out


def test_a_drifting_question_twice_means_the_codes_own_question_is_said():
    e, out = spoken(["How about its driving experience or features?", "Does that sound like a car you'd enjoy driving?"])
    assert out[-1] == "What else would you like to know about it?" and e.reply_writer.backend.calls == 2


def test_a_good_rewording_is_spoken_as_it_is():
    e, out = spoken(["Is there anything else you'd like to hear about it?"])
    assert out[-1] == "Is there anything else you'd like to hear about it?" and e.reply_writer.backend.calls == 1


def test_a_drift_then_a_good_second_try_is_accepted():
    e, out = spoken(["How about its driving experience?", "Anything else you'd like to know about it?"])
    assert out[-1] == "Anything else you'd like to know about it?" and e.prefix_stats["continuation"] == "accepted-on-retry"


def test_the_instruction_says_to_ask_only_the_given_question():
    from dealer_core.agent.engine import CONTINUE_SYSTEM

    assert "EXACTLY the question" in CONTINUE_SYSTEM and "not bring up any topic" in CONTINUE_SYSTEM
