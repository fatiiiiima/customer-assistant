"""Kokoro's pieces: a short first piece so the first sound comes quickly, then clauses."""
from dealer_core.voice.kokoro_tts import first_piece_split


def test_the_first_piece_is_short_and_nothing_is_lost():
    text = "The RAV4 is a 2021 in white, with 33,000 kilometres on it, at 98,500 dirhams."
    pieces = first_piece_split(text)
    assert len(pieces[0].split()) <= 7 and " ".join(pieces) == text
    assert first_piece_split("Want to book a test drive?") == ["Want to book a test drive?"]
