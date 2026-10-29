"""The owner's review of full conversations on PR 25 (29 Oct, docs/ARCH_REVIEW.md section 30b): the fillers ("Okay,", "So,", "Right,") at the start of 17 of 30
replies are off; "a family SUV under 100,000" got the Kia Carnival (a van at 126,000); and the model added opinions the prompt forbids ("making it great for long
family trips"). The cut-reply repair ("and here is the full picture.") is tested in test_cut_reply_repair.py."""
import sys
from pathlib import Path

import pytest

from dealer_core.config import load_config
from dealer_core.dialogue.writer import strip_opinion

sys.path.insert(0, str(Path(__file__).parent))


def test_the_fillers_are_off_by_default():
    assert load_config(environ={}).dialogue.openers is False


@pytest.mark.parametrize("said, kept", [
    ("It gets about 4.9 litres per hundred kilometres, which is efficient for a hybrid SUV, making it great for long family trips.",
     "It gets about 4.9 litres per hundred kilometres."),
    ("Right, it gets about 4.9 litres per hundred kilometres, which is efficient for a hybrid SUV and helps with daily family use.",
     "Right, it gets about 4.9 litres per hundred kilometres."),
    ("Yes, it's a good family car with 5 seats, a 580-litre boot, and all-wheel drive for better traction on different surfaces.",
     "Yes, it's a good family car with 5 seats, a 580-litre boot, and all-wheel drive."),
    ("Sure, the interior has black leather upholstery with a sporty feel and a panoramic roof for great views.",
     "Sure, the interior has black leather upholstery and a panoramic roof."),
    ("So, it has a 3.0-litre V6 twin-turbo engine producing 340 horsepower, known for strong performance and responsive acceleration.",
     "So, it has a 3.0-litre V6 twin-turbo engine producing 340 horsepower."),
    ("The interior is black leather and sits five, offering a refined and sporty feel.", "The interior is black leather and sits five."),     # the replay after the fix
])
def test_an_opinion_added_to_a_fact_is_cut_and_the_fact_stays(said, kept):
    assert strip_opinion(said) == kept


@pytest.mark.parametrize("said", ["It has a 2.5-litre engine, all-wheel drive, and a full service history.", "It's black.", "Would you like to come and see it?",
                                  "It has a full service history, which has every stamp from the dealer, and a 6-month warranty.",
                                  "It's the 2016 Porsche Macan in black,", "It's accident-free and passed 146 of our 150 inspection checks."])
def test_a_sentence_of_facts_is_left_as_it_is(said):
    assert strip_opinion(said) == said


def test_the_model_s_opinion_is_not_spoken():
    from test_model_speaks import engine

    e, _ = engine("It uses about 9.8 litres per hundred kilometres, which is efficient for an SUV, making it great for long trips.")
    s = e.new_session()
    for t in ("Porsche.", "Yes, tell me about the Macan."):
        list(e.handle_stream(s, t))
    said = [x for x in e.handle_stream(s, "What about the fuel economy?") if isinstance(x, str)]
    assert said[0].startswith("It uses about 9.8 litres per hundred kilometres") and not ({"great", "efficient", "main thing"} & set(" ".join(said).replace(".", " ").split()))
    assert s.history[-1][1] == "It uses about 9.8 litres per hundred kilometres."                    # the reply ends there


def _say(e, s, text):
    import test_sweep_fixes as F

    return F.say(e, s, text)


def test_the_best_family_car_is_chosen_within_the_budget_and_type_asked_for():
    import test_sweep_fixes as F

    if not F.SITE.exists():
        pytest.skip("the website is not on this machine")
    e = F.SW.make_engine(load_config(environ={}, overlay=F.ROOT / "config" / "portal.yaml"), F.SITE, F.SW.TooSlow())
    s = e.new_session()
    intent, reply = _say(e, s, "Hi, I'm looking for a family SUV under 100,000.")
    car = e.svc.get_car(s.car)
    assert "Carnival" not in reply and car.body_type.value == "suv" and car.price <= 100000


def test_with_nothing_in_the_budget_no_car_over_it_is_offered():
    import test_sweep_fixes as F

    if not F.SITE.exists():
        pytest.skip("the website is not on this machine")
    e = F.SW.make_engine(load_config(environ={}, overlay=F.ROOT / "config" / "portal.yaml"), F.SITE, F.SW.TooSlow())
    s = e.new_session()
    _, reply = _say(e, s, "What's the best family SUV under 30,000?")
    assert s.car is None or e.svc.get_car(s.car).price <= 30000
    assert "Carnival" not in reply and "Tucson" not in reply
