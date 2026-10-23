"""Is the speech so far a complete answer to what the agent just asked? (Words, not length: 'Toyota' and 'Can I book' are both short.)"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from dealer_core.dialogue.states import State
from dealer_core.voice.answers import is_complete_answer

TZ = ZoneInfo("Asia/Dubai")
NOW = datetime(2030, 3, 4, 10, 0, tzinfo=TZ)
VOCAB = {"toyota": ["corolla", "camry", "land cruiser"], "mercedes": ["c 300", "gle 450"], "bmw": ["x5"]}


def check(text, last_agent, state=State.INTEREST):
    return is_complete_answer(text, last_agent, state, [], VOCAB, NOW, TZ)


@pytest.mark.parametrize("text,last_agent,state", [
    ("Toyota", "I have 4 cars. Could you tell me your budget or a make?", State.INTEREST),
    ("a white one", "Do you have a colour in mind?", State.INTEREST),
    ("any budget", "Could you tell me your budget?", State.INTEREST),
    ("under one fifty thousand", "Could you tell me your budget?", State.INTEREST),
    ("yes", "Would you like to book a test drive?", State.ANSWERING),
    ("no thanks", "Would you like to book a test drive?", State.ANSWERING),
    ("tomorrow at three in the afternoon", "Which day and time would suit you?", State.SLOT),
    ("Saturday", "Which day would suit you better?", State.SLOT),
    ("my number is zero five zero one two three four five six seven", "And your mobile number?", State.CONTACT),
    ("my name is Ahmed Khan", "Can I have your name and mobile number?", State.CONTACT),
])
def test_complete_answers(text, last_agent, state):
    assert check(text, last_agent, state)


@pytest.mark.parametrize("text,last_agent", [
    ("Can I book", "What are you looking for today?"),
    ("I'm looking for", "What are you looking for today?"),
    ("Tell me about", "Which one would you like to hear more about?"),
    ("How much is", "Would you like to book a test drive?"),
    ("My name is", "Can I have your name?"),
])
def test_unfinished_sentences_are_not_answers(text, last_agent):
    assert not check(text, last_agent, State.INTEREST)
