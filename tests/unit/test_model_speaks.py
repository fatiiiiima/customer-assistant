"""ARCH_PLAN Phase 3, behind `dialogue.model_speaks`: at the full level the model writes the reply from all of the car's facts, nothing is said before it, no closing
question is forced on it outside the booking steps, and every sentence passes agent/guards.py (the facts of this turn only; fail closed). docs/ARCH_REVIEW.md section 23."""
import sys
from datetime import timezone
from pathlib import Path

import pytest

from dealer_core.agent import guards
from dealer_core.agent.engine import AgentEngine
from dealer_core.agent.governor import QualityGovernor
from dealer_core.app import ROOT, build_app
from dealer_core.config import load_config
from dealer_core.dialogue.states import State
from dealer_core.dialogue.writer import ReplyWriter, verify_sentence

sys.path.insert(0, str(Path(__file__).parent))
from test_not_repeating import NOW, SITE, Scripted  # noqa: E402

INVITING = ("test drive", "come and see", "visit", "in person", "take a look", "book")


class Model:
    """Writes `reply` for every model-written turn and remembers what it was asked."""

    def __init__(self, reply: str):
        self.reply, self.asked = reply, []

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        self.asked.append(user)
        for i in range(0, len(self.reply), 7):
            yield self.reply[i:i + 7]

    def lead_in(self, *a, **k):
        return "none"


def engine(reply: str, speaks: bool = True) -> tuple[AgentEngine, Model]:
    if SITE is None:
        pytest.skip("the website is not on this machine")
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=SITE, classifier=Scripted(), reply_mode="conversational", clock=lambda: NOW.astimezone(timezone.utc),
                    lead_backend=Scripted())
    model = Model(reply)
    e = AgentEngine(app.service, Scripted(), ReplyWriter(model, temperature=None))
    e.governor = QualityGovernor("full")                       # the level where the model writes the whole reply
    e.openers, e.model_speaks = True, speaks
    return e, model


def macan_turn(e: AgentEngine, text: str = "Yes, talk about the Macan.") -> list[str]:
    s = e.new_session()
    list(e.handle_stream(s, "Porsche."))
    out = list(e.handle_stream(s, text))
    return [x for x in out if isinstance(x, str)], s


ENGINE = "It has a 3.0-litre V6 twin-turbo engine with 340 horsepower."


# ------------------------------------------------------------------------------------------------ what the model is given, and what is not added
def test_the_model_sees_the_whole_spec_sheet_and_ends_its_reply_its_own_way():
    e, model = engine(ENGINE)
    said, _ = macan_turn(e)
    assert said == [ENGINE]                                                    # no "Sure," before it, no "Would you like to come and see it?" after it
    prompt = model.asked[-1]
    assert "Everything on the spec sheet" in prompt and "horsepower" in prompt and "litre boot" in prompt and "previous owner" in prompt
    assert "not waiting for an answer" in prompt


def test_without_the_flag_the_full_level_is_as_before():
    """An opener first; the model sees only the car's line, so its engine sentence (340 horsepower is not in those facts) is refused and the code's wording and
    closing question are said."""
    e, _ = engine(ENGINE, speaks=False)
    said, _ = macan_turn(e)
    assert said[0] == "Sure," and ENGINE not in said and said[-1].endswith("?")


def test_the_aspect_asked_about_is_named_to_the_model():
    e, model = engine("The boot holds 500 litres.")
    said, s = macan_turn(e)
    list(e.handle_stream(s, "and the boot?"))
    assert "The caller is asking about: boot" in model.asked[-1]


def test_what_the_caller_already_heard_is_named_to_the_model():
    e, model = engine(ENGINE)
    said, s = macan_turn(e)
    list(e.handle_stream(s, "tell me more about it"))
    assert "already heard about its" in model.asked[-1]


# ------------------------------------------------------------------------------------------------ the guard: fail closed
@pytest.mark.parametrize("reply, why", [
    ("It has a V6 engine with 410 horsepower.", "number"),                     # a figure the facts do not give
    ("It's a 2023 model with a V6 engine.", "number"),                         # a year the facts do not give
    ("It has a V6 engine, a bit like the camry.", "car name"),                 # another car, not in this turn's facts (lower case: the writer's capitals check misses it)
])
def test_a_sentence_with_a_fact_the_turn_did_not_give_is_not_said(reply, why):
    e, _ = engine(reply)
    said, _ = macan_turn(e)
    assert reply not in said and said                                          # the code's own wording is said instead
    assert e.fidelity["leaks"] >= 1 and e.fidelity["fell_back"] == 1           # stopped by the guard (once per attempt: a reply with nothing usable is asked for again)


def test_a_figure_that_is_true_of_another_attribute_is_a_known_limit():
    """The guard checks values, not what they are attached to: 500 is the Macan's boot in litres, so "500 horsepower" passes it. Recorded, not hidden."""
    e, _ = engine("It has a V6 engine with 500 horsepower.")
    said, _ = macan_turn(e)
    assert said == ["It has a V6 engine with 500 horsepower."] and e.fidelity["leaks"] == 0


def test_a_faithful_reply_passes_the_guard():
    e, _ = engine(ENGINE)
    macan_turn(e)
    assert e.fidelity == {"sentences": 1, "leaks": 0, "replies": 1, "fell_back": 0}


# ------------------------------------------------------------------------------------------------ which questions stay
def test_a_booking_step_keeps_its_question_and_a_clarifying_question_stays():
    e, _ = engine(ENGINE)
    s = e.new_session()
    s.state = State.SLOT
    core, awaiting, _ = e._speaking_input(s, "yes", {"core": "Sure. What day would you like to come in?", "facts": [], "awaiting": True})
    assert core.endswith("What day would you like to come in?") and awaiting
    s.state = State.ANSWERING
    core, awaiting, _ = e._speaking_input(s, "the black one", {"core": "Which car do you mean?", "facts": [], "awaiting": True})
    assert core == "Which car do you mean?" and awaiting                       # nothing else to say: the question is the reply
    core, awaiting, facts = e._speaking_input(s, "ok", {"core": "It has 340 horsepower. Would you like to come and see it?", "facts": [], "awaiting": True})
    assert core == "It has 340 horsepower." and not awaiting and "end on a statement" in facts


def test_a_booking_in_progress_or_made_is_named_so_the_model_does_not_offer_or_confirm():
    e, _ = engine(ENGINE)
    s = e.new_session()
    s.offered_slots = [NOW]
    assert "do not say it is booked" in e._speaking_input(s, "ok", {"core": "Fine.", "facts": [], "awaiting": False})[2]


# ------------------------------------------------------------------------------------------------ the guard itself (a fixed fact list: nothing outside it may be said)
FACTS = ("- 2016 Porsche Macan S: black, 91,000 kilometres, 149,000 dirhams, automatic gearbox, petrol, GCC spec\n"
         "It has a 3.0-litre V6 twin-turbo engine with 340 horsepower. It seats 5, with a 500-litre boot.")
NAMES = {"porsche", "macan", "toyota", "camry", "rav4", "land", "cruiser", "nissan", "patrol"}


@pytest.mark.parametrize("sentence", [
    "It's the 2016 Macan in black, with 91,000 kilometres on it.",
    "It's 149,000 dirhams.",
    "The boot holds 500 litres and it seats five.",
    "It makes 340 horsepower from a twin-turbo V6.",
    "It has a GCC-spec petrol engine.",
])
def test_faithful_sentences_pass(sentence):
    assert guards.leaks(sentence, FACTS, NAMES) == []


@pytest.mark.parametrize("sentence, kind", [
    ("It's 139,000 dirhams.", "number"), ("It's done 19,000 kilometres.", "number"), ("It's a 2019 model.", "number"),
    ("It makes 400 horsepower.", "number"), ("It comes in white too.", "colour"), ("It's like the Camry.", "car name"),
    ("It's bigger than a land cruiser.", "car name"), ("It does nought to a hundred in 5.2 seconds.", "number"),
    ("It costs one hundred and forty thousand dirhams.", "number"),
])
def test_leaks_are_caught(sentence, kind):
    found = guards.leaks(sentence, FACTS, NAMES)
    assert found and kind in found[0], found


def test_the_guard_is_stricter_than_the_sentence_check_about_the_history():
    """verify_sentence accepts a figure said earlier in the call; the guard does not: it is not in this turn's facts."""
    history = "AGENT: The Camry is 74,500 dirhams."
    sentence = "It's 74,500 dirhams."
    assert verify_sentence(sentence, "and the price?", "", FACTS, history) == []
    assert guards.leaks(sentence, FACTS, NAMES)


def test_the_stock_names_are_taken_from_the_inventory():
    names = guards.car_names({"Toyota": ["Land Cruiser", "RAV4"], "BMW": ["X5", "3 Series"]})
    assert {"toyota", "land", "cruiser", "rav4", "bmw", "x5"} <= names and "series" not in names and "3" not in names


# ------------------------------------------------------------------------------------------------ forbidden promises (the plan's suite): the writer's check and the guard together
PROMISES = ["I'll hold it for you until tomorrow.", "I can reserve it for you.", "We can give you a discount.", "It's booked for you.", "Your test drive is confirmed.",
            "I guarantee you'll love it.", "We'll deliver it to your door.", "We can arrange finance for you.", "I'll make sure nobody else buys it.",
            "It's in immaculate condition.", "There's a free service included.", "I promise it's the best deal in Dubai.", "We'll throw in free tinting.",
            "I can get my manager to lower the price.", "It comes with a full warranty.", "You can return it within a week.", "We'll take your old car in trade."]


@pytest.mark.parametrize("sentence", PROMISES)
def test_no_promise_the_facts_do_not_make_is_said(sentence):
    """Each is stopped by verify_sentence (the writer's check, 16 of 17) or by the guard (the 17th: a return policy with no 'I' or 'we' and no digit)."""
    assert verify_sentence(sentence, "tell me about it", "", FACTS, "") or guards.leaks(sentence, FACTS, NAMES)


def test_a_commercial_term_the_facts_give_may_be_said():
    facts = FACTS + "\nIt's had 2 previous owners, and comes with an AGMC warranty of 6 months or 10,000 km."
    assert guards.leaks("It comes with an AGMC warranty of 6 months.", facts, NAMES) == []
    assert guards.leaks("You can return it within a week.", facts, NAMES)
