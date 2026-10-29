"""What the voice is given: amounts, times, phone numbers, dates and car codes are rewritten into words the voice says correctly."""
import pytest

from dealer_core.voice.speech_text import speech_text
from dealer_core.voice.tts import pronounce

CASES = [
    ("It's AED 98,500, which is fixed.", "It's ninety eight thousand five hundred dirhams, which is fixed."),
    ("It costs 62,500 AED.", "It costs sixty two thousand five hundred dirhams."),
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
    ("28,000 km", "twenty eight thousand kilometres"),
]


@pytest.mark.parametrize("written,spoken", CASES)
def test_rewrite(written, spoken):
    assert speech_text(written) == spoken


def test_numbers_as_words_option():
    assert speech_text("That is 209,000 dirhams.", numbers_as_words=True) == "That is two hundred and nine thousand dirhams."


def test_the_brand_letters_still_work_after_the_rewrite():
    assert "eigh gee em see" in pronounce("Welcome to AGMC. It's AED 98,500.") and "dirhams" in pronounce("It's AED 98,500.")


def test_ordinary_words_and_plain_numbers_are_untouched():
    assert speech_text("I have 12 cars, a 2021 model.") == "I have 12 cars, a 2021 model."
    assert speech_text("under 150,000 dirhams, a 2021 model") == "under one hundred and fifty thousand dirhams, a 2021 model"      # amounts in words (29 Oct)


@pytest.mark.parametrize("piece, said", [
    ("There's the BMW 320i at 119,000 dirhams,", "There's the BMW three twenty i at one hundred and nineteen thousand dirhams,"),
    ("the BMW 530i at 189,000,", "the BMW five thirty i at one hundred and eighty nine thousand,"),
    ("the BMW M3 at 289,000. Which", "the BMW M three at two hundred and eighty nine thousand. Which"),
    ("with 91,000 kilometres on it,", "with ninety one thousand kilometres on it,"),
    ("It's the 2016 Porsche Macan.", "It's the 2016 Porsche Macan."),
])
def test_an_amount_is_given_to_the_voice_in_words(piece, said):
    """Owner's call, 29 Oct: "119,000" in a piece ending on a comma was said "one hundred nineteen, zero zero zero" (the voice's own number reading breaks on the
    trailing comma), while 209,000 at the end of a sentence was fine. Amounts with thousands separators are now always given as words; years are left alone."""
    assert speech_text(piece) == said
