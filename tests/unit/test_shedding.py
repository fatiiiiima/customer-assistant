"""When the model server is overwhelmed, turns skip it (load shedding) or stop waiting for it (deadline); the code decides and speaks at once."""
import random

from dealer_core.agent.governor import QualityGovernor

from .test_scale import NEEDS_THE_MODEL, Scripted, agent


def overloaded(decide_ms=1500.0, seed=1):
    g = QualityGovernor("auto", budget_ms=700, shed_after_ms=600, rng=random.Random(seed))
    g.record_decide(decide_ms)
    return g


def test_the_share_of_turns_that_skip_the_model_grows_with_the_overload_and_is_zero_when_it_is_fast():
    def share(ms):
        g = overloaded(ms)
        return sum(not g.model_allowed() for _ in range(2000)) / 2000

    assert share(400) == 0.0 and 0.1 < share(700) < 0.25 and 0.85 < share(1500) <= 0.92 and share(5000) <= 0.92
    assert QualityGovernor("auto", shed_after_ms=0).model_allowed()                     # shedding off


def test_a_shed_turn_is_decided_and_spoken_by_code_without_touching_the_model():
    e, b = agent(decide_ms=None)
    e.governor = overloaded(5000)                                                     # 90% shed
    shed = 0
    for _ in range(20):
        asked = len(b.prompts)
        r = e.handle(e.new_session(), NEEDS_THE_MODEL)                               # a line only the model could route
        if len(b.prompts) == asked:
            shed += 1
            assert "shed, decided by code" in r.debug.note and r.reply.endswith("?")
    assert 14 <= shed < 20                                                           # most skip it; a few still measure it


def test_with_nothing_to_go_on_a_shed_turn_asks_a_guiding_question_and_a_car_in_focus_is_answered():
    e, b = agent(decide_ms=None)
    e.governor = overloaded(5000, seed=3)
    s = e.new_session()
    replies = {e.handle(e.new_session(), NEEDS_THE_MODEL).reply for _ in range(10)}
    assert any("make, type of car or budget" in r or "a make, a type of car, or a budget" in r for r in replies)
    e.handle(s, "tell me about the Land Cruiser")
    tools = {e.handle(s, "would it handle the sand dunes well").debug.intent for _ in range(5)}
    assert "get_car" in tools


def test_a_decision_that_misses_the_deadline_is_replaced_by_the_codes_own():
    class Times(Scripted):
        def lead_in(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
            if "TOOL:" in user:
                self.prompts.append(user)
                raise TimeoutError("no decision within the deadline")
            return "ok"

    e, b = agent(decide_ms=None, backend=Times())
    e.governor = QualityGovernor("auto", shed_after_ms=600, rng=random.Random(1))
    r = e.handle(e.new_session(), NEEDS_THE_MODEL)
    assert len(b.prompts) == 1 and "model too slow" in r.debug.note and "make, type of car or budget" in r.reply or "a make, a type of car" in r.reply


def test_shedding_stops_by_itself_when_the_model_recovers():
    g = overloaded(3000)
    for _ in range(60):
        if g.model_allowed():
            g.record_decide(150)                                                     # the turns that still go to it find it fast again
    assert sum(not g.model_allowed() for _ in range(200)) == 0
