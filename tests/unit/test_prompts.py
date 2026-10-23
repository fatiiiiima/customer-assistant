from dealer_core.dialogue.states import State
from dealer_core.voice.prompts import CONTACT, CONFIRM, DAY_TIME, stt_prompt

G = {"makes": ["Toyota", "Kia"], "models": ["Camry"], "places": ["Dubai"], "common names": ["Ahmed", "Fatima"]}


def test_the_hint_follows_the_question():
    assert stt_prompt(State.SLOT, "I can do today at 11 or tomorrow at 9. Which works better for you?", G) == DAY_TIME
    assert stt_prompt(State.ANSWERING, "Would you like to book a test drive for it? What day suits you?", G) == DAY_TIME
    assert stt_prompt(State.CONTACT, "Thanks. And your mobile number?", G) == CONTACT and stt_prompt(State.CONTACT, "Can I have your name?", G) == ""
    assert stt_prompt(State.CONFIRM, "Is that right?", G) == CONFIRM
    default = stt_prompt(State.INTEREST, "What are you looking for today?", G)
    assert "Toyota" in default and "tomorrow" in default and "Ahmed" not in default      # names only when a name is expected
