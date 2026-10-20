import pytest

from dealer_core.dialogue.states import Intent, State
from dealer_core.nlu.consistency import repair


@pytest.mark.parametrize("state", [State.ANSWERING, State.NEGOTIATING])
def test_a_named_amount_turns_a_lowest_price_question_into_an_offer(state):
    assert repair(Intent.ASK_LOWEST_PRICE, state, "can you do 120,000?")[0] is Intent.MAKE_OFFER
    assert repair(Intent.ASK_LOWEST_PRICE, state, "would one hundred and ten thousand work")[0] is Intent.MAKE_OFFER


def test_a_real_lowest_price_question_is_left_alone():
    assert repair(Intent.ASK_LOWEST_PRICE, State.NEGOTIATING, "what's the best you can do") == (Intent.ASK_LOWEST_PRICE, None)
    assert repair(Intent.ASK_LOWEST_PRICE, State.NEGOTIATING, "is it the 2021 model with 62000 km")[1] is None   # year and km are not prices


def test_other_intents_are_never_touched():
    assert repair(Intent.MAKE_OFFER, State.NEGOTIATING, "can you do 120,000")[1] is None
    assert repair(Intent.CONFIRM_YES, State.CONFIRM, "yes 120,000")[1] is None
