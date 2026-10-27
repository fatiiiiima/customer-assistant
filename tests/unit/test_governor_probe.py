"""27 Oct, the Macan call: the governor tried the model-written ("full") level on the first caller turn after a start, and again every 8th turn, to find out whether the
model had got faster. On a laptop that turn took 3.3 s to the first sentence and the caller heard "So," and nothing more. Now a background probe measures the model; a
caller's turn is never the test."""
import asyncio
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from dealer_core.agent.engine import AgentEngine
from dealer_core.agent.governor import QualityGovernor
from dealer_core.app import ROOT, build_app
from dealer_core.config import load_config
from dealer_core.dialogue.states import Intent
from dealer_core.nlu.backend import IntentResult
from dealer_core.server.app import probe_loop

NOW = datetime(2026, 10, 27, 11, 0, tzinfo=ZoneInfo("Asia/Dubai"))


# ------------------------------------------------------------------------------------------------ the governor
def test_an_unmeasured_model_is_played_safe_when_a_probe_is_running():
    g = QualityGovernor("auto", budget_ms=700)
    assert g.choose() == "full"                        # no probe: the old behaviour (text chat, tests)
    g = QualityGovernor("auto", budget_ms=700)
    g.probed = True
    assert [g.choose() for _ in range(5)] == ["prefix"] * 5
    assert g.snapshot()["probed"] is True


def test_a_caller_turn_is_never_used_to_test_the_slow_level_by_default():
    g = QualityGovernor("auto", budget_ms=700)
    for _ in range(5):
        g.record_full(1800)                              # the model is slow
    assert [g.choose() for _ in range(40)] == ["prefix"] * 40          # idle, 40 turns: it never tries "full" on a caller
    assert g.snapshot()["turns"]["full"] == 0


def test_exploring_on_live_turns_is_still_available_when_asked_for():
    g = QualityGovernor("auto", budget_ms=700, explore_every=3)
    for _ in range(5):
        g.record_full(1800)
    assert "full" in [g.choose() for _ in range(6)]


def test_probes_are_the_way_back_up():
    g = QualityGovernor("auto", budget_ms=700)
    g.probed = True
    g.record_probe(3300)                                 # the first probe: slow
    assert g.choose() == "prefix"
    g.record_probe(300)
    g.record_probe(300)
    g.record_probe(300)
    g.record_probe(300)                                  # the model has room again
    assert g.snapshot()["full_first_ms"] < 700 and g.choose() == "full"


def test_a_probe_counts_for_more_than_a_live_turn():
    a, b = QualityGovernor("auto"), QualityGovernor("auto")
    a.record_full(1000)
    b.record_probe(1000)
    a.record_full(200)
    b.record_probe(200)
    assert b.snapshot()["full_first_ms"] < a.snapshot()["full_first_ms"]       # 600 vs 760


# ------------------------------------------------------------------------------------------------ the measuring request
class Scripted:
    name = "scripted"

    def classify(self, *a, **k):
        return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

    def lead_in(self, *a, **k):
        return "none"

    def lead_in_stream(self, *a, **k):
        yield "ok."


class SlowWriter:
    """Stands in for the reply writer: a first sentence after `delay` seconds."""

    def __init__(self, delay=0.05, fail=False, nothing=False):
        self.delay, self.fail, self.nothing, self.inputs, self.closed = delay, fail, nothing, [], False
        self.last_stream = {"stopped": "", "sentences": 2}                  # what the real writer leaves behind for the engine's trace

    def stream(self, w, temperature=None):
        self.inputs.append(w)
        try:
            if self.fail:
                raise RuntimeError("model down")
            time.sleep(self.delay)
            if not self.nothing:
                yield "It's a white RAV4 with 33,000 kilometres."
                yield "Would you like to hear more?"
        finally:
            self.closed = True


def engine(writer=None, governor=True):
    cfg = load_config(environ={})
    app = build_app(cfg, inventory="demo", classifier=Scripted(), reply_mode="conversational", clock=lambda: NOW.astimezone(timezone.utc), lead_backend=Scripted())
    e = AgentEngine(app.service, Scripted(), writer if writer is not None else app.engine.reply_writer)
    if governor:
        e.governor = QualityGovernor("auto", budget_ms=700)
    return e


def test_the_probe_times_a_first_sentence_and_tells_the_governor():
    w = SlowWriter(delay=0.05)
    e = engine(w)
    ms = e.probe_model()
    assert 40 <= ms < 1500 and e.governor.snapshot()["full_first_ms"] == round(ms)
    assert w.closed                                          # only the first sentence was wanted: the model was stopped
    sent = w.inputs[0]
    assert sent.history and sent.facts and "kilometres" in sent.core and sent.caller            # the shape of a real turn


def test_a_probe_leaves_no_trace_in_any_call():
    e = engine(SlowWriter(0))
    s = e.new_session()
    before = (list(s.history), s.car, s.state)
    e.probe_model()
    assert (list(s.history), s.car, s.state) == before


def test_a_priming_probe_is_not_recorded():
    e = engine(SlowWriter(0.01))
    assert e.probe_model(record=False) is not None
    assert e.governor.snapshot()["full_first_ms"] == 0


@pytest.mark.parametrize("writer", [SlowWriter(fail=True), SlowWriter(nothing=True), None])
def test_a_probe_that_cannot_run_is_skipped_not_an_error(writer):
    e = engine(writer if writer is not None else SimpleNamespace(), governor=True)
    if writer is None:
        e.reply_writer = None
    assert e.probe_model() is None and e.governor.snapshot()["full_first_ms"] == 0


def test_without_stock_there_is_nothing_to_probe_with():
    e = engine(SlowWriter(0))
    e.svc.inventory.search = lambda *a, **k: []
    assert e.probe_model() is None


# ------------------------------------------------------------------------------------------------ the first caller turn after a start
def test_the_first_model_written_turn_after_a_start_is_not_a_test():
    e = engine(SlowWriter(0))
    e.governor.probed = True                                  # the server has a probe running; nothing is measured yet
    s = e.new_session()
    list(e.handle_stream(s, "I'm looking for a Toyota"))
    s.car = s.shown[0].stock_id
    list(e.handle_stream(s, "tell me about the Camry"))
    assert e.governor.snapshot()["turns"]["full"] == 0 and e.last_level == "prefix"


def test_after_a_fast_probe_the_next_turn_is_model_written():
    e = engine(SlowWriter(0.001))
    e.governor.probed = True
    e.probe_model()
    s = e.new_session()
    list(e.handle_stream(s, "I'm looking for a Toyota"))
    s.car = s.shown[0].stock_id
    list(e.handle_stream(s, "tell me about the Camry"))
    assert e.last_level == "full"


# ------------------------------------------------------------------------------------------------ the loop in the server
class Gov:
    def __init__(self, in_flight=0):
        self.in_flight = in_flight

    def snapshot(self):
        return {"in_flight": self.in_flight}


class Eng:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def probe_model(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")


def run_loop(eng, gov, rounds, on_sleep=None, tries=10):
    slept = []

    async def sleep(s):
        slept.append(s)
        if on_sleep:
            on_sleep(len(slept))

    asyncio.run(probe_loop(eng, gov, 20.0, sleep=sleep, rounds=rounds, tries=tries))
    return slept


def test_the_loop_probes_every_interval_when_the_model_is_idle():
    eng = Eng()
    slept = run_loop(eng, Gov(0), rounds=3)
    assert eng.calls == 3 and slept == [20.0, 20.0, 20.0]


def test_the_loop_waits_for_a_gap_between_caller_turns_and_does_not_interrupt_one():
    eng, gov = Eng(), Gov(2)
    slept = run_loop(eng, gov, rounds=1, on_sleep=lambda n: setattr(gov, "in_flight", 0) if n == 4 else None)
    assert eng.calls == 1 and slept == [20.0, 1.0, 1.0, 1.0]       # waited three seconds for a free moment


def test_the_loop_gives_up_on_a_round_when_the_model_never_gets_free():
    eng = Eng()
    slept = run_loop(eng, Gov(3), rounds=2, tries=4)
    assert eng.calls == 0 and slept.count(20.0) == 2


def test_a_failing_probe_does_not_stop_the_loop():
    eng = Eng(fail=True)
    run_loop(eng, Gov(0), rounds=3)
    assert eng.calls == 3


# ------------------------------------------------------------------------------------------------ settings
def test_the_defaults_and_the_environment():
    d = load_config(environ={}).dialogue
    assert d.explore_every == 0 and d.probe_every_s == 20.0
    d = load_config(environ={"VA__DIALOGUE__PROBE_EVERY_S": "0", "VA__DIALOGUE__EXPLORE_EVERY": "8"}).dialogue
    assert d.probe_every_s == 0 and d.explore_every == 8
    app = build_app(load_config(environ={}), inventory="demo", classifier=Scripted(), reply_mode="conversational", engine="agent", lead_backend=Scripted())
    assert app.engine.governor.explore_every == 0
