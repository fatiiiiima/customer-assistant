"""The three numbers that bound how long a caller waits on the model live in ONE section (`latency`); the old names still work (Phase 2)."""
import pytest

from dealer_core.app import ROOT
from dealer_core.config import LatencyCfg, load_config
from dealer_core.config import Config


def test_the_defaults_are_the_values_the_old_places_had():
    c = load_config(environ={}).latency
    assert (c.model_budget_ms, c.decide_deadline_ms, c.first_sentence_deadline_ms) == (700.0, 1200.0, 1500)


def test_the_old_places_no_longer_hold_them():
    cfg = load_config(environ={})
    assert not hasattr(cfg.dialogue, "model_budget_ms")
    assert not hasattr(cfg.voice, "decide_deadline_ms") and not hasattr(cfg.voice, "first_sentence_deadline_ms")


@pytest.mark.parametrize("old, new, value, expect", [
    ("VA__DIALOGUE__MODEL_BUDGET_MS", "model_budget_ms", "450", 450.0),
    ("VA__VOICE__DECIDE_DEADLINE_MS", "decide_deadline_ms", "800", 800.0),
    ("VA__VOICE__FIRST_SENTENCE_DEADLINE_MS", "first_sentence_deadline_ms", "0", 0),
])
def test_an_environment_variable_with_the_old_name_still_sets_it(old, new, value, expect):
    assert getattr(load_config(environ={old: value}).latency, new) == expect


def test_the_new_name_wins_when_both_are_given():
    cfg = load_config(environ={"VA__VOICE__DECIDE_DEADLINE_MS": "800", "VA__LATENCY__DECIDE_DEADLINE_MS": "300"})
    assert cfg.latency.decide_deadline_ms == 300.0


def test_an_overlay_file_with_the_old_names_still_works(tmp_path):
    overlay = tmp_path / "o.yaml"
    overlay.write_text("dialogue:\n  model_budget_ms: 350\nvoice:\n  first_sentence_deadline_ms: 900\n", encoding="utf-8")
    cfg = load_config(environ={}, overlay=overlay)
    assert cfg.latency.model_budget_ms == 350 and cfg.latency.first_sentence_deadline_ms == 900


def test_a_misspelt_key_is_still_refused():
    with pytest.raises(Exception):
        load_config(environ={"VA__LATENCY__DECIDE_DEADLINE": "800"})


def test_the_numbers_reach_what_uses_them():
    from dealer_core.voice.pipeline import apply_voice_profile
    from dealer_core.app import build_app
    from tests.unit.test_slow_model import Brain

    cfg = load_config(environ={"VA__LATENCY__MODEL_BUDGET_MS": "400", "VA__LATENCY__DECIDE_DEADLINE_MS": "900", "VA__LATENCY__FIRST_SENTENCE_DEADLINE_MS": "1100"},
                      overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="demo", classifier=Brain(), reply_mode="conversational", lead_backend=Brain(), engine="agent")
    apply_voice_profile(app, cfg)
    assert app.engine.governor.budget_ms == 400
    assert app.engine.decide_deadline_s == 0.9
    assert app.engine.reply_writer.first_deadline_s == 1.1


def test_the_section_says_where_its_values_come_from():
    doc = LatencyCfg.__doc__
    assert "PHASE0_RUNBOOK" in doc and "CLOSED" in doc and "laptop" in doc
