"""The call cap follows the measured reply times: down fast when replies are slow, up slowly when they are fast and the cap is what limits."""
from dealer_core.server.admission import AdmissionController


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def controller(**kw):
    clock = Clock()
    return AdmissionController(16, target_s=1.0, clock=clock, **kw), clock


def feed(a, clock, seconds, n=20, every=0.2):
    for _ in range(n):
        clock.t += every
        a.record(seconds)


def test_slow_replies_cut_the_cap_and_running_calls_are_not_counted_twice():
    a, clock = controller()
    assert a.cap == 16 and a.accepts(15) and not a.accepts(16)
    feed(a, clock, 1.6)                                       # slow: 1.6 s
    clock.t += 5
    assert a.adjust(16) == 12                                 # 16 * 0.75
    clock.t += 5
    assert a.adjust(16) == 12                                 # the slow replies before the cut do not count again
    feed(a, clock, 1.4)
    clock.t += 5
    assert a.adjust(12) == 9 and a.cuts == 2
    assert not a.accepts(9) and a.accepts(8)                  # new calls wait; the 9 running are untouched


def test_fast_replies_raise_the_cap_only_while_it_is_what_limits_and_never_past_the_ceiling():
    a, clock = controller(start_calls=10)
    feed(a, clock, 0.5)
    clock.t += 5
    assert a.adjust(3) == 10                                  # only 3 calls: the cap is not what limits, keep it
    clock.t += 5
    assert a.adjust(10) == 11                                 # full and fast: one more
    for _ in range(20):
        feed(a, clock, 0.5, n=3)
        clock.t += 5
        a.adjust(a.cap)
    assert a.cap == 16                                        # the ceiling (memory) holds


def test_between_the_target_and_the_headroom_it_holds_and_with_few_replies_it_waits():
    a, clock = controller()
    feed(a, clock, 0.9)                                       # under 1 s but above 0.8 s: hold
    clock.t += 5
    assert a.adjust(16) == 16
    b, clock2 = controller()
    feed(b, clock2, 3.0, n=5)                                 # 5 slow replies are not enough to judge
    clock2.t += 5
    assert b.adjust(16) == 16


def test_cuts_that_help_are_kept_down_to_where_replies_meet_the_target_never_below_the_minimum():
    a, clock = controller(min_calls=2)
    for _ in range(12):
        feed(a, clock, 0.6 + 0.06 * a.cap)                    # this worker's load is the cause: fewer calls, faster replies
        clock.t += 5
        a.adjust(a.cap)
    assert 2 <= a.cap <= 6 and 0.6 + 0.06 * a.cap <= 1.0 and a.reverts == 0
    b, clock2 = controller(min_calls=2)
    for _ in range(12):
        feed(b, clock2, 0.2 + 0.3 * b.cap)                    # even 2 calls are too slow: it stops at the minimum
        clock2.t += 5
        b.adjust(b.cap)
    assert b.cap == 2


def test_a_cut_that_does_not_make_replies_faster_is_undone_and_cutting_pauses():
    a, clock = controller()
    feed(a, clock, 1.5)                                       # slow whatever the load (a slow model server, a slow network)
    clock.t += 5
    assert a.adjust(16) == 12
    feed(a, clock, 1.5)
    clock.t += 5
    assert a.adjust(12) == 16 and a.reverts == 1              # not faster after the cut: turning callers away saved nothing
    snap = a.snapshot()
    assert snap["slow_regardless"] and snap["cuts"] == 1      # for an alert: the hardware, not the load, misses the target
    feed(a, clock, 1.5)
    clock.t += 5
    assert a.adjust(16) == 16                                 # no new cut during the pause
    clock.t += 60
    feed(a, clock, 1.5)
    clock.t += 5
    assert a.adjust(16) == 12                                 # after it, it tries again


def test_the_worker_refuses_past_its_measured_cap_and_tells_the_router_that_cap():
    import json
    import time
    import types

    from fastapi.testclient import TestClient

    from dealer_core.agent.governor import QualityGovernor
    from dealer_core.server.app import ServerState, create_server
    from dealer_core.shared.state import MemoryStore, WorkerRegistry

    reg = WorkerRegistry(MemoryStore())
    state = ServerState(app=types.SimpleNamespace(engine=types.SimpleNamespace(governor=QualityGovernor("auto"))), parts=None, cfg=None,
                        max_calls=16, registry=reg, worker_id="w1", public_url="ws://w1/voice")
    state.admission = AdmissionController(16, start_calls=3)         # its replies were slow: it takes 3 calls, not 16
    state.active = 3
    with TestClient(create_server(state)) as client:
        deadline = time.time() + 2
        while not reg.workers() and time.time() < deadline:
            time.sleep(0.02)
        w = reg.workers()[0]
        assert w["max"] == 3 and w["ceiling"] == 16                   # the router sees the measured cap
        with client.websocket_connect("/voice", headers={"origin": "http://localhost:8080"}) as ws:
            assert json.loads(ws.receive_text()) == {"type": "busy"}
        health = client.get("/health").json()
        assert health["cap"] == 3 and health["admission"]["max_calls"] == 16


def test_the_answer_time_skips_a_filler_said_first():
    from pipecat.frames.frames import BotStartedSpeakingFrame, TranscriptionFrame, TTSSpeakFrame, VADUserStoppedSpeakingFrame

    from dealer_core.voice.status import TurnClock

    def run(script):
        clock, seen = Clock(), {}
        tc = TurnClock(0.2, lambda total, parts: seen.setdefault("first", round(total, 2)), clock=clock,
                       on_answer=lambda s: seen.setdefault("answer", round(s, 2)), fillers=frozenset({"Let me check."}))
        for at, frame, stage in script:
            clock.t = at
            tc.see(frame, stage)
        return seen

    stop = VADUserStoppedSpeakingFrame(stop_secs=0.2)
    heard = TranscriptionFrame("any red cars", "u", "t")
    # "Let me check." at 0.4 s, its sound at 0.45 s; the answer is ready at 1.6 s (a slow model)
    seen = run([(1.0, stop, "vad"), (1.2, heard, "stt"), (1.4, TTSSpeakFrame("Let me check."), "brain"), (1.45, BotStartedSpeakingFrame(), ""),
                (2.6, TTSSpeakFrame("There's the Altima at 52,500 dirhams."), "brain")])
    assert seen == {"first": 0.65, "answer": 1.85}                    # first sound 0.65 s; the answer itself 1.85 s (+ the voice's 0.05 s)
    seen = run([(1.0, stop, "vad"), (1.2, heard, "stt"), (1.5, TTSSpeakFrame("There's the Altima."), "brain"), (1.6, BotStartedSpeakingFrame(), "")])
    assert seen == {"first": 0.8, "answer": 0.8}                      # no filler: the answer is the first sound
