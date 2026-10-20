import pytest

from dealer_core.nlu.grounding import ground_slots
from dealer_core.nlu.numbers import canon_numbers, int_to_words, normalise


@pytest.mark.parametrize("text,expected", [
    ("my budget is fifty five thousand dirhams", ["55000"]),
    ("I can offer 95,000", ["95000"]),
    ("a twenty nineteen model with forty thousand kilometres", ["2019", "40000"]),
    ("TWENTY TWENTY ONE", ["2021"]),
    ("one hundred and sixty two thousand nine hundred", ["162900"]),
    ("zero five zero one two three four five six seven", ["0501234567"]),
    ("0501234567", ["0501234567"]),
    ("+971 50 444 3322", ["971504443322"]),
    ("twenty five", ["25"]),
    ("twenty one thousand five hundred", ["21500"]),
])
def test_canon_numbers(text, expected):
    assert canon_numbers(text) == expected


def test_int_to_words_roundtrip():
    for n in (7, 19, 20, 55, 100, 101, 999, 1000, 55000, 162900, 1_250_000):
        assert canon_numbers(int_to_words(n)) == [str(n)]


def test_normalise_spells_digits():
    assert normalise("It costs 12,500!") == "it costs twelve thousand five hundred"


def test_grounding_drops_invented_values():
    u = "can you do forty eight thousand"
    kept = ground_slots(u, {"offer_amount": 48000, "topic": "warranty", "make": "toyota", "max_km": 0})
    assert kept == {"offer_amount": 48000}


def test_grounding_keeps_supported_values():
    u = "do you offer a warranty on the Toyota Camry"
    assert ground_slots(u, {"topic": "warranty", "make": "Toyota", "model": "Camry"}) == {
        "topic": "warranty", "make": "Toyota", "model": "Camry"}


def test_grounding_phone_digits_only():
    assert ground_slots("my number is zero five zero one two three four five six seven", {"phone": "050-123-4567"}) == {"phone": "050-123-4567"}
    assert ground_slots("my number is zero five zero one two three four five six seven", {"phone": "0509999999"}) == {}
