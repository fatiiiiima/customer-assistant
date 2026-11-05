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


def test_observe_never_steps_down_and_never_sheds():
    g = QualityGovernor("observe", budget_ms=700, shed_after_ms=450, max_in_flight_model=2)
    for _ in range(8):
        g.record_full(3000)                          # a very slow model
        g.record_decide(5000)
    with g.using_model(), g.using_model(), g.using_model():   # and more turns using it than the cap
        assert all(g.choose() == "full" for _ in range(20))
    assert all(g.model_allowed() for _ in range(200))
    assert g.has_room()
    snap = g.snapshot()
    assert snap["turns"] == {"full": 20, "prefix": 0, "code": 0} and snap["shed"] == 0
    assert snap["full_first_ms"] > 2000 and snap["decide_ms"] > 3000        # it still measures: the metrics and alerts read these


def test_observe_counts_the_first_sentence_in_the_expected_wait():
    g = QualityGovernor("observe", budget_ms=700)
    g.record_decide(400)
    g.record_full(1500)                              # slower than the budget, but nothing replaces it: the caller waits for it
    assert g.expected_wait_ms() > 1800
