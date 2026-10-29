"""28 Oct review (fix 11): every offer of test-drive times began with a piece of speech made on the spot ("Today I can do", "The soonest is today:"), about 0.3-0.55 s to the first sound.
The first piece is the first four words, so where only the day varies it can be made before the first call. Measured on this Mac with the real voice: 318-556 ms before, 0 ms after."""
import inspect

import pytest

from dealer_core.agent.wording import offer_slot_variants
from dealer_core.voice.kokoro_tts import KokoroOV, first_piece_split
from dealer_core.voice.pipeline import slot_offer_openings

TIMES = ["2 in the afternoon or 5 in the evening", "9 in the morning or 2 in the afternoon", "11 in the morning or 3 in the afternoon", "10:30 in the morning or 4 in the afternoon"]
DAYS = ["today", "tomorrow", "on Monday", "on Tuesday", "on Wednesday", "on Thursday", "on Friday", "on Saturday", "on Sunday"]


def covered():
    return {first_piece_split(s)[0] for s in slot_offer_openings()}


@pytest.mark.parametrize("day", DAYS)
@pytest.mark.parametrize("times", TIMES)
def test_the_opening_of_every_offer_that_does_not_start_with_a_time_is_ready_in_advance(day, times):
    kept = covered()
    for sentence in offer_slot_variants(day, times):
        sentence = sentence.split("?")[0].split(". ")[0] + "."                         # the sentence that is spoken first (the question after it is a phrase of its own)
        first = first_piece_split(sentence)[0]
        assert first in kept or any(ch.isdigit() for ch in first), f"{first!r} is neither ready in advance nor started by a time"


def test_the_ones_the_review_asked_for_are_there():
    kept = covered()
    assert {"Today I can do", "Tomorrow I can do", "The soonest is today:", "The soonest is tomorrow:"} <= kept


def test_an_opening_with_a_time_in_its_first_four_words_is_not_rendered():
    kept = covered()
    assert not any(any(ch.isdigit() for ch in piece) for piece in kept)               # "Today I've got 9" would be a different piece for every hour


def test_only_the_first_piece_of_each_opening_is_rendered_not_the_times_after_it():
    rendered = []

    class Stub(KokoroOV):
        def __init__(self):                                                           # (no model: just the counting)
            self._cache, self._cache_size, self.opening_first, self._pinned = {}, 300, False, {}

        def _say(self, piece):
            rendered.append(piece)

    Stub().preload([], first_only=["Today I can do 9 in the morning or 2 in the afternoon.", "The soonest is today: 9 in the morning or 2 in the afternoon."])
    assert rendered == ["Today I can do", "The soonest is today:"]


def test_the_phrases_given_without_the_option_are_rendered_whole_as_before():
    rendered = []

    class Stub(KokoroOV):
        def __init__(self):
            self._cache, self._cache_size, self.opening_first, self._pinned = {}, 300, False, {}

        def _say(self, piece):
            rendered.append(piece)

    Stub().preload(["Sure, the Camry is a 2020 hybrid in brown, with 52,000 kilometres."])
    assert len(rendered) >= 2 and rendered[0] == "Sure,"


def test_the_voice_is_loaded_with_the_openings():
    assert "first_only=slot_offer_openings(tts.pieces)" in inspect.getsource(__import__("dealer_core.voice.pipeline", fromlist=["x"]).load_engines)


def test_the_wording_itself_is_unchanged():
    out = offer_slot_variants("today", "2 in the afternoon or 5 in the evening")
    assert out == ["Today I can do 2 in the afternoon or 5 in the evening. Which suits you better?", "Today I've got 2 in the afternoon or 5 in the evening. Which works for you?",
                   "The soonest is today: 2 in the afternoon or 5 in the evening. Which would you prefer?"]


@pytest.mark.parametrize("day", DAYS)
@pytest.mark.parametrize("times", TIMES)
def test_with_openings_first_the_first_piece_of_every_offer_is_still_ready_in_advance(day, times):
    """B (voice.opening_first, 29 Oct) cuts sentences differently; the openings rendered in advance are cut the same way, so they still match."""
    from dealer_core.voice.kokoro_tts import opening_first_split

    kept = {opening_first_split(s)[0] for s in slot_offer_openings(opening_first_split)}
    for sentence in offer_slot_variants(day, times):
        sentence = sentence.split("?")[0].split(". ")[0] + "."
        first = opening_first_split(sentence)[0]
        assert first in kept or any(ch.isdigit() for ch in first), f"{first!r} is neither ready in advance nor started by a time"
