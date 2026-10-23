"""What the voice is given: amounts, times, phone numbers, dates and car codes are rewritten into words the voice says correctly."""
import pytest

from dealer_core.voice.speech_text import speech_text
from dealer_core.voice.tts import pronounce

CASES = [
    ("It's AED 98,500, which is fixed.", "It's 98,500 dirhams, which is fixed."),
    ("It costs 62,500 AED.", "It costs 62,500 dirhams."),
    ("tomorrow at 4:00 PM", "tomorrow at four P M"),
    ("Saturday at 10:30 AM", "Saturday at ten thirty A M"),
    ("at 9:05 am", "at nine oh five A M"),
    ("24 October", "the twenty fourth of October"),
    ("1 March", "the first of March"),
    ("Is your number 050 123 4567?", "Is your number zero five zero, one two three, four five six seven?"),
    ("Call +971 50 123 4567.", "Call zero five zero, one two three, four five six seven."),
    ("the BMW X5", "the BMW X five"),
    ("the BMW 320i", "the BMW three twenty i"),
    ("the BMW M3", "the BMW M three"),
    ("the Mazda CX-5", "the Mazda C X five"),
    ("the Toyota RAV4", "the Toyota RAV four"),
    ("the Mercedes GLC 300", "the Mercedes G L C 300"),
    ("the Land Cruiser GXR", "the Land Cruiser G X R"),
    ("122 hp", "122 horsepower"),
    ("28,000 km", "28,000 kilometres"),
]


@pytest.mark.parametrize("written,spoken", CASES)
def test_rewrite(written, spoken):
    assert speech_text(written) == spoken


def test_numbers_as_words_option():
    assert speech_text("That is 209,000 dirhams.", numbers_as_words=True) == "That is two hundred and nine thousand dirhams."


def test_the_brand_letters_still_work_after_the_rewrite():
    assert "eigh gee em see" in pronounce("Welcome to AGMC. It's AED 98,500.") and "dirhams" in pronounce("It's AED 98,500.")


def test_ordinary_words_and_plain_numbers_are_untouched():
    assert speech_text("I have 12 cars under 150,000 dirhams, a 2021 model.") == "I have 12 cars under 150,000 dirhams, a 2021 model."
