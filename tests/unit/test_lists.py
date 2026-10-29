"""29 Oct, the owner: "for Toyota it lists three and says plus one more; for the makes six or seven, then plus 9 more". Now one or two left over are named, and a longer
rest is offered and read out on a yes. docs/ARCH_REVIEW.md section 28."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import test_sweep_fixes as F  # noqa: E402


def call(lines):
    from dealer_core.config import load_config

    if not F.SITE.exists():
        import pytest
        pytest.skip("the website is not on this machine")
    e = F.SW.make_engine(load_config(environ={}, overlay=F.ROOT / "config" / "portal.yaml"), F.SITE, F.SW.TooSlow())
    s = e.new_session()
    return e, s, [F.say(e, s, x) for x in lines]


def test_four_toyotas_are_all_named():
    e, s, turns = call(["I'm looking for a Toyota.", "Any model and any budget is fine."])
    reply = turns[-1][1]
    assert all(m in reply for m in ("Corolla", "Camry", "RAV4", "Land Cruiser")) and "plus one more" not in reply and " more." not in reply and len(s.shown) == 4


def test_the_makes_offer_the_rest_and_a_yes_names_them():
    e, s, turns = call(["What cars do you have?", "yes"])
    assert turns[0][1].endswith("or shall I name the others?") and "10 other makes" in turns[0][1]
    assert turns[1][0] == "browse_makes" and "Volkswagen" in turns[1][1] and "Hyundai" in turns[1][1]


def test_a_long_list_offers_the_rest_and_each_yes_reads_the_next():
    e, s, turns = call(["I'm looking for an SUV.", "any budget", "yes"])
    first, rest = turns[1][1], turns[2][1]
    assert "and 9 more" in first and first.endswith("or shall I go through the rest?")
    assert turns[2][0] == "search_cars" and "Tucson" not in rest and "Patrol" in rest


def test_a_yes_to_something_else_is_not_taken_as_the_rest():
    e, s, turns = call(["I'm looking for a Toyota.", "Any model and any budget is fine.", "yes"])
    assert "Land Cruiser" not in turns[-1][1] or turns[-1][0] != "search_cars"           # the Toyota list offered no rest
