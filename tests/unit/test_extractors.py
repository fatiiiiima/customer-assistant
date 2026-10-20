import pytest

from dealer_core.models import SearchFilters
from dealer_core.nlu import extractors as ex


@pytest.fixture
def vocab(demo_inventory):
    return demo_inventory.vocabulary()


@pytest.mark.parametrize("text,expected", [
    ("I'm looking for an SUV under 150 thousand dirhams", {"body_type": "suv", "max_price": 150000}),
    ("any SUV under 150k", {"body_type": "suv", "max_price": 150000}),
    ("do you have a Toyota Land Cruiser from 2020", {"make": "toyota", "model": "land cruiser", "year_min": 2020}),
    ("Toyode Land Cruiser please", {"make": "toyota", "model": "land cruiser"}),
    ("a sedan with less than eighty thousand kilometres", {"body_type": "sedan", "max_km": 80000}),
    ("Nissan please", {"make": "nissan"}),
    ("the Patrol", {"make": "nissan", "model": "patrol"}),
    ("something nice", {}),
    ("hello there", {}),
])
def test_search_filters(vocab, text, expected):
    assert ex.search_filters(text, vocab) == expected


@pytest.mark.parametrize("text,amount", [
    ("can you do 135,000", 135000), ("how about one hundred and twenty thousand dirhams", 120000), ("I'll pay 115k", 115000),
    ("it has 62000 km and I want it for 130000", 130000), ("2021 model", None), ("my number is 0501234567", None),
    ("what's your best price", None),
])
def test_offer_amount(text, amount):
    assert ex.offer_amount(text) == amount


def test_selection_by_ordinal_name_and_comparison(demo_inventory):
    shown = [c.view() for c in demo_inventory.search(SearchFilters(make="toyota", body_type="suv", max_price=150_000), 3)]
    ids = [c.stock_id for c in shown]
    assert ids == ["S1053", "S1042"]                                  # cheapest first: Prado, then Land Cruiser
    assert ex.selection("the first one", shown) == ids[0] and ex.selection("tell me about the second one", shown) == ids[1]
    assert ex.selection("the last one", shown) == ids[-1]
    assert ex.selection("the Land Cruiser", shown) == "S1042" and ex.selection("the prado", shown) == "S1053"
    assert ex.selection("the cheaper one", shown) == min(shown, key=lambda c: c.price).stock_id
    assert ex.selection("the Patrol", shown) is None and ex.selection("hello", shown) is None and ex.selection("the first one", []) is None


@pytest.mark.parametrize("text,keys", [
    ("how many kilometres has it done", ["km"]), ("what colour is it and is it GCC spec", ["color", "spec"]),
    ("has it had any accidents", ["accident"]), ("does it have a service history", ["service"]),
    ("is it automatic or manual", ["transmission"]), ("which branch is it at", ["branch"]), ("hello", []),
])
def test_attributes_asked(text, keys):
    assert ex.attributes_asked(text) == keys


@pytest.mark.parametrize("text,name,phone", [
    ("My name is Mohammed Al Mansoori and my number is zero five zero one two three four five six seven", "Mohammed Al Mansoori", "+971501234567"),
    ("Aisha, 050 123 4567", "Aisha", "+971501234567"),
    ("this is Khalid Rashid, zero five five nine eight seven six five four three", "Khalid Rashid", "+971559876543"),
    ("it's 0501234567", None, "+971501234567"),
    ("my name is Sara", "Sara", None),
    ("yes", None, None), ("my number is one two three", None, None),
])
def test_contact(text, name, phone):
    c = ex.contact(text)
    assert (c.name, c.phone) == (name, phone)


def test_contact_reports_digits_that_are_not_a_valid_uae_number():
    c = ex.contact("my number is one two three four five six seven eight nine")
    assert c.phone is None and c.heard_digits == "123456789"


def test_wants_others():
    assert ex.wants_others("show me other models") and ex.wants_others("I'm not looking for something like that")
    assert not ex.wants_others("the first one")
