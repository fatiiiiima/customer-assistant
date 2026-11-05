"""2 Nov, the owner's live call: "Any model, any budget." after "I'm looking for a Toyota" was answered with a Tesla Model Y. The caller's pair of words "model any" was matched to the
stock's "model y" as a string (88%); every later turn then carried the Tesla as the car in focus (ARCH_REVIEW section 89)."""
from dealer_core.nlu import extractors as ex

VOCAB = {"tesla": ["model 3", "model y"], "toyota": ["camry", "land cruiser", "rav4"], "land rover": ["range rover sport", "range rover velar"], "bentley": ["bentayga"]}


def found(text):
    return ex.find_vehicle(ex.tokens_of(text), VOCAB)


def test_the_word_model_is_not_a_car():
    for text in ("Any model, any budget.", "any model any budget", "I don't mind the model, any year", "which model is it"):
        assert found(text) == {}, text


def test_the_cars_that_really_are_said_are_still_found_including_slips_in_a_long_name():
    assert found("tell me about the model y") == {"make": "tesla", "model": "model y"}
    assert found("the Tesla model three") == {"make": "tesla", "model": "model 3"} or found("the Tesla model three").get("make") == "tesla"
    assert found("what about the range rover sport") == {"make": "land rover", "model": "range rover sport"}
    assert found("a range rover sprot please").get("model") == "range rover sport"                # a mis-hearing of one word of a long name
    assert found("the bentayga").get("model") == "bentayga" and found("the bentyga").get("model") == "bentayga"


def test_several_words_must_each_be_close_not_only_the_whole_string():
    assert ex._same_words("model any", "model y") is False
    assert ex._same_words("range rover sprot", "range rover sport") is True
    assert ex._same_words("model a", "model y") is False and ex._same_words("rav 4", "rav4") is False      # different number of words: not this rule
