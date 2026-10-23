"""Spoken prices, the way people say them (including the colloquial 'one fifty thousand' = 150,000, which used to read as 51,000)."""
import pytest

from dealer_core.nlu import extractors as ex
from dealer_core.nlu.numbers import canon_numbers

CASES = [
    ("under one fifty thousand", 150_000),
    ("a hundred and fifty k", 150_000),
    ("under 150k", 150_000),
    ("fifty-five thousand five hundred", 55_500),
    ("fifty five thousand five hundred", 55_500),
    ("under 55,500 dirhams", 55_500),
    ("one hundred and fifty thousand", 150_000),
    ("two twenty thousand", 220_000),
    ("two hundred and nine thousand", 209_000),
    ("twenty one thousand", 21_000),          # tens then ones: unchanged
    ("around eighty grand", 80_000),
    ("three hundred thousand", 300_000),
]


@pytest.mark.parametrize("text,amount", CASES)
def test_spoken_amount_is_read_correctly(text, amount):
    assert ex.offer_amount(text) == amount
    assert ex.search_filters(text, {}).get("max_price") == amount


def test_kilometres_are_not_prices():
    assert ex.search_filters("under fifty k km", {}).get("max_km") == 50_000
    assert "max_price" not in ex.search_filters("under fifty k km", {})


def test_a_phone_number_and_a_year_are_not_amounts():
    assert ex.offer_amount("my number is zero five zero one two three four five six seven") is None
    assert ex.search_filters("a twenty twenty one model", {}) == {"year_min": 2021}
    assert canon_numbers("two thousand") == ["2000"]
