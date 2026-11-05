"""The cross-talk test script (scripts/crosstalk_test.py): that callers' details are all different, and that a mix-up is told from a recogniser's mistake."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("crosstalk_test", ROOT / "scripts" / "crosstalk_test.py")
xt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xt)

CALLERS = xt.make_callers(24, "t")


def test_every_caller_has_details_nobody_else_has():
    assert len({c["name"] for c in CALLERS}) == 24 and len({c["phone"] for c in CALLERS}) == 24 and len({c["call_id"] for c in CALLERS}) == 24
    assert all(c["phone"].startswith("05599") and len(c["phone"]) == 10 for c in CALLERS)
    assert len({tuple(c["lines"][3:4]) for c in CALLERS}) == 24                    # the day and time too, so the server has no reason to refuse a booking


def test_too_many_callers_is_refused():
    try:
        xt.make_callers(55, "t")
    except ValueError:
        return
    raise AssertionError("55 callers cannot all have a different day and time")


def test_a_booking_with_my_number_is_mine_whatever_the_server_writes_and_another_callers_details_are_crossed():
    me, other = CALLERS[0], CALLERS[1]
    plus = "+971" + me["phone"][1:]                                                # the server saves 0559912345 as +971559912345
    assert xt.classify_booking({"name": me["name"], "phone": plus}, me, CALLERS) == "own"
    assert xt.classify_booking({"name": "Layla Muley", "phone": plus}, me, CALLERS) == "own_name_misheard"      # the recogniser's mistake with a name, the number is mine
    assert xt.classify_booking({"name": other["name"], "phone": plus}, me, CALLERS) == "crossed"
    assert xt.classify_booking({"name": me["name"], "phone": "+971" + other["phone"][1:]}, me, CALLERS) == "crossed"


def test_a_recogniser_mistake_is_not_a_mixup():
    me = CALLERS[0]
    assert xt.classify_booking({"name": "Sarah Con", "phone": "0559900000"}, me, CALLERS) == "misheard"


def test_another_callers_details_are_found_as_a_name_digits_or_spoken_digits():
    me, other = CALLERS[0], CALLERS[1]
    spoken = " ".join(xt.DIGITS[int(d)] for d in other["phone"])
    assert xt.mentions(f"Booked for {other['name']}.", other)
    assert xt.mentions(f"Your number {other['phone']} is saved", other)
    assert xt.mentions("Booked, +971" + other["phone"][1:], other)                                          # as the server writes it
    assert xt.mentions(f"Is it {spoken}?", other)
    assert xt.mentions("Hello, do you have any Toyota SUVs?", other) is None
    assert xt.check_texts([f"Thanks {other['name']}"], me, CALLERS) and not xt.check_texts([f"Thanks {me['name']}"], me, CALLERS)     # my own details are not a problem
