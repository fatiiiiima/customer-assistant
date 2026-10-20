from dealer_core.dialogue.states import Intent, State
from dealer_core.eval.datasets import Example
from dealer_core.eval.scoring import score
from dealer_core.eval.stats import gate_verdict, mcnemar_exact, paired_compare


def ex(utt, intent, slots=None):
    return Example("x", "t", State.NEGOTIATING, "S1", utt, intent, slots or {})


def test_exact_needs_intent_and_all_slots():
    e = ex("can you do forty eight thousand", Intent.MAKE_OFFER, {"offer_amount": 48000})
    assert score(e, {"intent": "make_offer", "slots": {"offer_amount": 48000}}, ground=True).exact
    assert not score(e, {"intent": "make_offer", "slots": {}}, ground=True).exact
    assert not score(e, {"intent": "confirm_yes", "slots": {"offer_amount": 48000}}, ground=True).exact


def test_grounding_removes_invented_slot_that_would_fail_exact():
    e = ex("yes that works for me", Intent.CONFIRM_YES)
    pred = {"intent": "confirm_yes", "slots": {"offer_amount": 0}}
    assert not score(e, pred, ground=False).exact
    assert score(e, pred, ground=True).exact


def test_invalid_prediction_is_not_valid():
    assert not score(ex("hi", Intent.GREET), None, ground=True).valid
    assert not score(ex("hi", Intent.GREET), {"slots": {}}, ground=True).valid


def test_mcnemar_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert abs(mcnemar_exact(0, 8) - 2 / 256) < 1e-12


def test_paired_compare_identical_systems_is_zero():
    flags = [True, False, True, True] * 25
    r = paired_compare(flags, flags)
    assert r.diff_pts == 0 and r.ci_low == 0 and r.ci_high == 0
    assert "NO LARGE DROP" in gate_verdict(r)


def test_gate_verdict_confirms_a_big_consistent_drop():
    base = [True] * 100
    cand = [True] * 80 + [False] * 20
    r = paired_compare(cand, base)
    assert r.diff_pts == -20 and r.ci_high < -5
    assert gate_verdict(r).startswith("DROP CONFIRMED")


def test_gate_verdict_inconclusive_when_interval_spans_threshold():
    base = [True] * 40 + [False] * 10
    cand = [True] * 37 + [False] * 13        # -6 points on only 50 items: noise is large
    r = paired_compare(cand, base)
    assert gate_verdict(r).startswith("INCONCLUSIVE")
