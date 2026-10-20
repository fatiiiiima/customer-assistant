from pathlib import Path

import pytest

from dealer_core.dialogue.states import STATE_CHANGING, Intent, State
from dealer_core.eval.datasets import load_json_set
from dealer_core.nlu.rules import RuleEngine, load_enabled_intents
from dealer_core.repos.seed import TEMPLATES

ROOT = Path(__file__).resolve().parents[2]
VOCAB: dict[str, list[str]] = {}
for _make, _model, _, _ in TEMPLATES:
    VOCAB.setdefault(_make.lower(), []).append(_model.lower())
REAL = load_enabled_intents(ROOT / "config" / "nlu_gates.yaml")


def engine(enabled=None):
    return RuleEngine(lambda: VOCAB, set(Intent) if enabled is None else enabled)


def test_shipped_gates_open_no_state_changing_intent():
    assert REAL and not (REAL & STATE_CHANGING)


def test_a_state_changing_gate_cannot_be_opened_without_evidence(tmp_path):
    f = tmp_path / "g.yaml"
    f.write_text("enabled: [greet, confirm_yes]\nevidence: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="confirm_yes"):
        load_enabled_intents(f)
    f.write_text("enabled: [greet, confirm_yes]\nevidence: {confirm_yes: {n: 50, lower_bound: 0.93}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="98%"):
        load_enabled_intents(f)
    f.write_text("enabled: [greet, confirm_yes]\nevidence: {confirm_yes: {n: 212, lower_bound: 0.982}}\n", encoding="utf-8")
    assert Intent.CONFIRM_YES in load_enabled_intents(f)
    f.write_text("enabled: [not_an_intent]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_enabled_intents(f)


def test_closed_gate_defers_to_the_llm_instead_of_falling_through():
    e = engine(REAL)
    assert e.classify(State.CONFIRM, "S1042", "yes that's right") is None                 # confirm_yes is closed
    assert e.classify(State.NEGOTIATING, "S1042", "can you do 120000") is None            # make_offer is closed
    assert e.classify(State.INTEREST, None, "I'm looking for an SUV under 150 thousand dirhams").intent is Intent.SEARCH_CARS
    assert e.classify(State.ANSWERING, "S1042", "how many kilometres has it done").intent is Intent.ASK_CAR_INFO
    # an angry utterance must NOT slip through to a weaker harmless rule when the angry gate is closed
    assert e.classify(State.ANSWERING, "S1042", "this is useless, how many kilometres has it done") is None


def test_rules_are_state_gated():
    e = engine()
    assert e.classify(State.INTEREST, None, "how many kilometres has it done") is None    # ask_car_info is not valid with no car
    assert e.classify(State.ANSWERING, "S1042", "how many kilometres has it done").intent is Intent.ASK_CAR_INFO


@pytest.mark.parametrize("name,min_cov,min_prec", [("dev", 0.90, 0.95), ("heldout", 0.85, 0.95), ("fresh", 0.75, 0.90)])
def test_regression_against_the_labelled_sets_with_all_gates_open(name, min_cov, min_prec):
    """Guards the port of the Phase 0b prototype: same coverage and precision on the same sets (not a tuning target)."""
    e = engine()
    ex = load_json_set(ROOT / "data" / "nlu_sets" / f"{name}.json")
    hits = [(x, e.classify(x.state, x.car, x.utterance)) for x in ex]
    fired = [(x, h) for x, h in hits if h]
    assert len(fired) / len(ex) >= min_cov
    assert sum(h.intent is x.gold_intent for x, h in fired) / len(fired) >= min_prec


def test_real_gates_never_answer_a_state_changing_intent():
    e = engine(REAL)
    for name in ("dev", "heldout", "fresh"):
        for x in load_json_set(ROOT / "data" / "nlu_sets" / f"{name}.json"):
            h = e.classify(x.state, x.car, x.utterance)
            assert h is None or h.intent not in STATE_CHANGING
