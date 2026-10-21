import pytest

from dealer_core.dialogue.states import Intent, State
from dealer_core.nlu.consistency import repair


@pytest.mark.parametrize("state", [State.ANSWERING, State.NEGOTIATING])
def test_a_named_amount_turns_a_lowest_price_question_into_an_offer(state):
    assert repair(Intent.ASK_LOWEST_PRICE, state, "can you do 120,000?")[0] is Intent.MAKE_OFFER
    assert repair(Intent.ASK_LOWEST_PRICE, state, "would one hundred and ten thousand work")[0] is Intent.MAKE_OFFER


def test_a_real_lowest_price_question_is_left_alone():
    assert repair(Intent.ASK_LOWEST_PRICE, State.NEGOTIATING, "what's the best you can do") == (Intent.ASK_LOWEST_PRICE, None)
    assert repair(Intent.ASK_LOWEST_PRICE, State.NEGOTIATING, "is it the 2021 model with 62000 km")[0] is not Intent.MAKE_OFFER   # year and km are not prices


def test_other_intents_are_never_touched():
    assert repair(Intent.MAKE_OFFER, State.NEGOTIATING, "can you do 120,000")[1] is None
    assert repair(Intent.CONFIRM_YES, State.CONFIRM, "yes 120,000")[1] is None


def test_price_intent_without_price_wording_is_not_a_price_question():
    from dealer_core.dialogue.states import Intent, State
    from dealer_core.nlu.consistency import repair

    assert repair(Intent.ASK_LOWEST_PRICE, State.NEGOTIATING, "bently")[0] is Intent.OUT_OF_SCOPE
    assert repair(Intent.ASK_LOWEST_PRICE, State.NEGOTIATING, "what's the best you can do")[0] is Intent.ASK_LOWEST_PRICE
    assert repair(Intent.ASK_LOWEST_PRICE, State.NEGOTIATING, "any room to move")[0] is Intent.ASK_LOWEST_PRICE


def test_unstocked_make_is_found_even_misspelled():
    from dealer_core.nlu import extractors as ex

    vocab = {"toyota": ["camry"], "nissan": ["patrol"]}
    for said in ("do you have a betley", "bently", "do you have a bently car"):
        assert ex.search_filters(said, vocab).get("make") == "bentley", said
    assert ex.search_filters("do you have a camry", vocab).get("make") == "toyota"


def test_angry_needs_a_sign_of_anger():
    from dealer_core.dialogue.states import Intent, State
    from dealer_core.nlu.consistency import repair

    assert repair(Intent.ANGRY, State.ANSWERING, "its so hot today")[0] is Intent.OUT_OF_SCOPE
    assert repair(Intent.ANGRY, State.ANSWERING, "ok whatever")[0] is Intent.OUT_OF_SCOPE
    for really_angry in ("this is useless", "you are wasting my time", "this is a joke", "i am so frustrated with you", "what a scam"):
        assert repair(Intent.ANGRY, State.ANSWERING, really_angry)[0] is Intent.ANGRY, really_angry


def test_an_amount_with_offer_wording_is_an_offer_even_if_the_model_says_car_question():
    from dealer_core.dialogue.states import Intent, State
    from dealer_core.nlu.consistency import repair

    for said in ("how about 125,000", "what about one hundred and twenty five thousand", "can you do 120000"):
        assert repair(Intent.ASK_CAR_INFO, State.NEGOTIATING, said, True)[0] is Intent.MAKE_OFFER, said
    assert repair(Intent.ASK_CAR_INFO, State.NEGOTIATING, "how many kilometres are on 2021 model", True)[0] is Intent.ASK_CAR_INFO
    assert repair(Intent.ASK_CAR_INFO, State.NEGOTIATING, "how about the colour", True)[0] is Intent.ASK_CAR_INFO         # no amount: not an offer
