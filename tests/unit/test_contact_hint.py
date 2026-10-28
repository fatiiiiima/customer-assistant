"""28 Oct, 13:19: at "Can I take your name and mobile number?" the caller said "No." and the recogniser wrote "050, 056, 056, 0", the digits it had been primed with, so the agent went on
to ask for a name. Measured with the base recogniser, synthetic speech and this room's noise: "No, thanks.", "Not now." and "My name is Omar." all came back as a run of "056"s with the digits
hint alone, and came back right with "no, yes, not now" added to it; whole phone numbers were recognised the same with both (14 of 14)."""
import pytest

from dealer_core.dialogue.states import State
from dealer_core.voice.prompts import CONTACT, DAY_TIME, stt_prompt
from dealer_core.voice.services import UNCLEAR, EchoGuard, is_short_digit_run

G = {"makes": ["Toyota"], "models": ["Camry"], "places": [], "common names": []}


def test_the_number_hint_also_allows_a_short_answer_that_is_not_a_number():
    assert CONTACT.startswith("050, 055, 056, 052, double five") and all(w in CONTACT for w in ("no", "yes", "not now"))
    assert "my name is" not in CONTACT                                                # still not primed with a phrase it would continue and loop on
    assert stt_prompt(State.CONTACT, "Can I take your name and mobile number?", G) == CONTACT


@pytest.mark.parametrize("text", ["050, 056, 056, 0", "0", "050", "050 056", "056, 056, 056, 056", "050, 0", "050, 057, 057, 057, 057", "050, 050, 056, 056, 056, 056, 056, 056, 056, 056, 056, 056, 0"])
def test_a_few_digits_or_a_repeated_group_are_not_a_phone_number(text):
    assert is_short_digit_run(text)


@pytest.mark.parametrize("text", ["", "No.", "no, thanks", "055 123 4567", "0551234567", "050 123 456 789", "my number is 055 123 4567", "double five", "055 555 5555", "056, 234, 5678", "050 123 4567"])
def test_these_are_left_alone(text):
    assert not is_short_digit_run(text)


class Engine:
    def __init__(self, first, second="No."):
        self.first, self.second, self.prompts = first, second, []
        self.name = "fake"

    def transcribe(self, pcm, prompt=None):
        self.prompts.append(prompt)
        return (self.second if prompt == "" else self.first), 10.0


def test_the_digits_written_for_a_short_answer_are_heard_again_without_the_hint():
    e = Engine("050, 056, 056, 0")
    g = EchoGuard(e, lambda: "Great, today at 3 PM it is. Can I take your name and mobile number?", lambda: "")
    text, ms = g.transcribe(b"\0\0", CONTACT)
    assert text == "No." and e.prompts == [CONTACT, ""] and ms == 20.0 and g.retries == 1


def test_a_real_number_is_passed_through_in_one_run():
    e = Engine("055 123 4567")
    g = EchoGuard(e, lambda: "And your mobile number?", lambda: "")
    assert g.transcribe(b"\0\0", CONTACT) == ("055 123 4567", 10.0) and e.prompts == [CONTACT] and g.retries == 0


def test_the_retry_happens_only_when_the_digits_hint_was_the_one_given():
    e = Engine("050 056")
    g = EchoGuard(e, lambda: "Which day suits you?", lambda: "")
    assert g.transcribe(b"\0\0", DAY_TIME)[0] == "050 056" and e.prompts == [DAY_TIME]


def test_nothing_on_the_second_listening_asks_the_caller_to_repeat():
    e = Engine("050, 056, 056, 0", second="")
    g = EchoGuard(e, lambda: "And your mobile number?", lambda: "")
    assert g.transcribe(b"\0\0", CONTACT)[0] == UNCLEAR
