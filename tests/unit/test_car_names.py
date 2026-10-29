"""Car names the caller says without the make, or in the plural (29 Oct, found in a barge-in call: after a Toyota search "do you have any BMWs?" and "tell me about the
X5" found no car, and the code went on with the Toyota search)."""
import sys
from pathlib import Path

import pytest

from dealer_core.nlu import extractors as ex

sys.path.insert(0, str(Path(__file__).parent))

VOCAB = {"bmw": ["320i", "530i", "m3", "x5"], "audi": ["a4", "q7"], "toyota": ["corolla", "camry", "rav4", "land cruiser"], "porsche": ["macan"]}


@pytest.mark.parametrize("said, make, model", [("the X5", "bmw", "x5"), ("tell me about the M3", "bmw", "m3"), ("the Q7", "audi", "q7"),
                                               ("any BMWs?", "bmw", None), ("Audis", "audi", None), ("Do you have any Porsche's?", "porsche", None),
                                               ("Toyotas", "toyota", None), ("the Camry", "toyota", "camry")])
def test_the_car_is_found(said, make, model):
    found = ex.find_vehicle(ex.tokens_of(said), VOCAB)
    assert found.get("make") == make and found.get("model") == model


@pytest.mark.parametrize("said", ["a four wheel drive", "is it a four door", "it has two owners", "x five", "yes"])
def test_ordinary_words_are_not_a_car(said):
    assert ex.find_vehicle(ex.tokens_of(said), VOCAB) == {}


def test_after_a_toyota_search_another_make_or_its_model_is_found():
    import test_sweep_fixes as F

    for line, want in (("Tell me about the X5.", "2019 BMW X5"), ("Do you have any BMWs?", "BMW")):
        e, s = F.at("search")                                                 # "I'm looking for a Toyota." was said
        _, reply = F.say(e, s, line)
        assert want in reply and "Corolla" not in reply
