"""The quality governor: full while the model is fast enough, prefix when it is slow, code when it is overloaded, and it looks again."""
from dealer_core.agent.governor import QualityGovernor


def test_full_while_fast_then_prefix_when_slow_then_back():
    g = QualityGovernor("auto", budget_ms=700, explore_every=3)
    assert g.choose() == "full"                      # nothing measured yet: try the best
    g.record_full(300)
    assert g.choose() == "full"
    for _ in range(5):
        g.record_full(1800)                          # the model got slow (a laptop, or peak load)
    assert g.choose() == "prefix"
    picks = [g.choose() for _ in range(4)]
    assert "full" in picks                           # it tries full again now and then
    for _ in range(8):
        g.record_full(250)                           # room again
    assert g.choose() == "full"


def test_code_when_too_many_turns_use_the_model():
    g = QualityGovernor("auto", budget_ms=700, max_in_flight_model=2)
    with g.using_model(), g.using_model():
        assert g.choose() == "code"
    assert g.choose() == "full"


def test_a_forced_level_is_kept():
    g = QualityGovernor("prefix")
    g.record_full(100)
    assert g.choose() == "prefix" and g.snapshot()["turns"]["prefix"] == 1
