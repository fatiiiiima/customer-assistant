"""Haggling (owner, 29 Oct; docs/ARCH_REVIEW.md section 29): the price never changes. The first push on a car gets the fixed price; from the second push on the
same car, cars near the price asked for are offered instead, each at its own fixed price. Counted per car. Website stock: the BMW X5 is 209,000 dirhams."""
import sys
from pathlib import Path

import pytest

from dealer_core.nlu import extractors as ex

sys.path.insert(0, str(Path(__file__).parent))


@pytest.mark.parametrize("text, price, want", [("can you do 150,000", 209000, 150000), ("come on can you do 150", 209000, 150000), ("150k?", 209000, 150000),
                                               ("would you take 120", 149000, 120000), ("I have 2 kids", 209000, None), ("can you do 5", 209000, None),
                                               ("what's your best price", 209000, None), ("it has 71,000 kilometres", 209000, None)])
def test_the_price_asked_for_is_read_from_the_words(text, price, want):
    assert ex.asked_price(text, price) == want


def _x5():
    import test_sweep_fixes as F

    e, s = F.at("price_q")
    assert e.svc.get_car(s.car).model == "X5"
    return F, e, s


def test_the_first_push_gets_the_fixed_price_alone():
    F, e, s = _x5()
    shown = list(s.shown)
    intent, reply = F.say(e, s, "Can you do 150,000?")
    assert intent == "get_car" and "209,000" in reply and "fixed" in reply
    assert s.haggles[s.car]["pushes"] == 1 and s.haggles[s.car]["offered"] == [] and s.shown == shown


def test_the_second_push_offers_cars_near_the_asked_price_and_the_price_stays():
    F, e, s = _x5()
    x5 = s.car
    F.say(e, s, "Can you do 150,000?")
    intent, reply = F.say(e, s, "Come on, can you do 150?")
    assert intent == "get_car" and s.car == x5 and e.svc.get_car(x5).price == 209000
    assert "209,000" in reply and "Q7" in reply and "Macan" in reply
    assert [c.model for c in s.shown] == ["Q7", "Macan"] and all(c.price < 209000 for c in s.shown)
    assert reply.endswith("Would you like to hear about one of them?")


def test_the_asked_price_is_remembered_and_each_push_names_new_cars():
    F, e, s = _x5()
    F.say(e, s, "Can you do 150,000?")
    F.say(e, s, "Come on, can you do 150?")
    _, reply = F.say(e, s, "What's your best price?")             # no amount: still about 150,000, and not the Q7 and Macan again
    assert "There's also" in reply and "Q7" not in reply and {c.model for c in s.shown} == {"Wrangler", "Patrol"}


def test_without_an_amount_the_cars_are_cheaper_than_this_one():
    F, e, s = _x5()
    F.say(e, s, "Any discount?")
    _, reply = F.say(e, s, "Can you knock some off?")
    assert "for less" in reply and s.shown and all(c.price < 209000 for c in s.shown)


def test_a_car_offered_can_be_picked_and_its_own_count_starts_at_one():
    F, e, s = _x5()
    F.say(e, s, "Can you do 150,000?")
    F.say(e, s, "Come on, can you do 150?")
    intent, _ = F.say(e, s, "Tell me about the first one.")
    assert intent == "get_car" and e.svc.get_car(s.car).model == "Q7"
    _, reply = F.say(e, s, "Any discount on it?")                # the first push on the Q7: its fixed price, no other cars
    assert "149,000" in reply and "Macan" not in reply and s.haggles[s.car]["pushes"] == 1


def test_asking_whether_the_price_is_good_is_not_a_push():
    F, e, s = _x5()
    F.say(e, s, "Can you do 150,000?")
    _, reply = F.say(e, s, "Is that a good price?")
    assert s.haggles[s.car]["pushes"] == 1 and "Q7" not in reply


def test_come_on_is_a_push_on_the_price_not_a_visit():
    F, e, s = _x5()
    intent, reply = F.say(e, s, "Come on, can you do 180?")
    assert intent == "get_car" and "209,000" in reply and s.state.value != "slot"


def test_the_cheapest_car_says_so_and_names_nothing_else():
    import test_sweep_fixes as F

    e, s = F.at("search")
    F.say(e, s, "Tell me about the Honda Civic.")
    F.say(e, s, "Any discount?")
    _, reply = F.say(e, s, "Can you do 45k?")
    assert "54,500" in reply and "lowest-priced car we have" in reply and "Corolla" not in reply


def test_the_count_survives_a_reconnect():
    from dealer_core.shared.state import restore, snapshot

    F, e, s = _x5()
    F.say(e, s, "Can you do 150,000?")
    fresh = e.new_session()
    restore(snapshot(s), fresh, e.svc)
    assert fresh.haggles == s.haggles
    _, reply = F.say(e, fresh, "Come on, can you do 150?")
    assert "Q7" in reply
