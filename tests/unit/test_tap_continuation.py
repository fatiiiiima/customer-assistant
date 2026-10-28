"""Tap mode used to close the microphone the moment the server decided the turn was over, so the rest of a sentence said after a long pause was never heard ("I'm looking
for a..." / "...Toyota"). Now the microphone stays open until the reply starts to play, and a caller who carries on in that window continues the SAME sentence."""
import asyncio
import threading

import pytest
from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InputAudioRawFrame, TranscriptionFrame, VADUserStartedSpeakingFrame,
                                   VADUserStoppedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection

from dealer_core.voice.brain import DealerBrain
from dealer_core.voice.micgate import ContinuationGuard, MicGate

DOWN = FrameDirection.DOWNSTREAM


def audio():
    return InputAudioRawFrame(audio=bytes(640), sample_rate=16000, num_channels=1)


def collect(proc):
    out = []

    async def push(frame, direction=DOWN):
        out.append(frame)
    proc.push_frame = push
    return out


# ------------------------------------------------------------------------------------------------ the gate
def test_the_microphone_stays_open_until_the_reply_starts_and_then_closes():
    async def go():
        gate = MicGate(start_open=True)
        out = collect(gate)
        told = []

        async def on_close():
            told.append("mic off")
        gate.hold_open_until_reply(5.0, on_close)
        await gate.process_frame(audio(), DOWN)
        assert gate.pending and gate.open and len(out) == 1                       # audio still passes while the reply is being prepared
        await gate.process_frame(BotStartedSpeakingFrame(), DOWN)
        await asyncio.sleep(0)
        assert not gate.open and not gate.pending and told == ["mic off"]
        await gate.process_frame(audio(), DOWN)
        assert len(out) == 2                                                      # (the bot frame itself); the audio after it is dropped
    asyncio.run(go())


def test_the_microphone_closes_by_itself_when_no_reply_starts():
    async def go():
        gate = MicGate(start_open=True)
        collect(gate)
        told = []

        async def on_close():
            told.append(1)
        gate.hold_open_until_reply(0.05, on_close)
        await asyncio.sleep(0.15)
        assert not gate.open and told == [1]
    asyncio.run(go())


def test_the_caller_speaking_again_keeps_it_open_and_cancels_the_closing():
    async def go():
        gate = MicGate(start_open=True)
        collect(gate)
        told = []

        async def on_close():
            told.append(1)
        gate.hold_open_until_reply(0.05, on_close)
        gate.cancel_pending()
        await asyncio.sleep(0.15)
        assert gate.open and not gate.pending and told == []
    asyncio.run(go())


def test_a_bot_starting_with_nothing_pending_changes_nothing():
    async def go():
        gate = MicGate(start_open=True)
        collect(gate)
        await gate.process_frame(BotStartedSpeakingFrame(), DOWN)
        assert gate.open
    asyncio.run(go())


def test_tapping_to_finish_ends_the_waiting():
    async def go():
        gate = MicGate(start_open=True)
        collect(gate)
        told = []

        async def on_close():
            told.append(1)
        gate.hold_open_until_reply(5.0, on_close)
        await gate.set_open(False)
        assert not gate.open and not gate.pending and told == []
    asyncio.run(go())


# ------------------------------------------------------------------------------------------------ the guard
class FakeBrain:
    def __init__(self, interrupt=True):
        self.calls, self.interrupt = [], interrupt

    async def take_back(self):
        self.calls.append("take_back")
        return self.interrupt

    def turn_ended(self):
        self.calls.append("turn_ended")


def guard_run(pending, interrupt=True):
    async def go():
        gate = MicGate(start_open=True)
        collect(gate)
        if pending:
            async def on_close():
                pass
            gate.hold_open_until_reply(5.0, on_close)
        brain = FakeBrain(interrupt)
        guard = ContinuationGuard(gate, brain)
        out, interrupted = collect(guard), []

        async def broadcast():
            interrupted.append(1)
        guard.broadcast_interruption = broadcast
        await guard.process_frame(VADUserStartedSpeakingFrame(start_secs=0.1), DOWN)
        return gate, brain, interrupted, out, guard
    return asyncio.run(go())


def test_speech_before_the_reply_is_heard_continues_the_turn():
    gate, brain, interrupted, out, guard = guard_run(pending=True)
    assert brain.calls == ["take_back"] and interrupted == [1] and not gate.pending and gate.open and guard.continued == 1
    assert [type(f).__name__ for f in out] == ["VADUserStartedSpeakingFrame"]             # the frame goes on to recognition as usual


def test_nothing_is_interrupted_when_nothing_was_on_its_way():
    _, brain, interrupted, _, _ = guard_run(pending=True, interrupt=False)
    assert brain.calls == ["take_back"] and interrupted == []


def test_speech_when_no_turn_is_waiting_is_an_ordinary_turn():
    gate, brain, interrupted, _, guard = guard_run(pending=False)
    assert brain.calls == [] and interrupted == [] and guard.continued == 0


def test_the_end_of_a_turn_clears_a_stale_wait_in_the_brain():
    async def go():
        brain = FakeBrain()
        guard = ContinuationGuard(MicGate(start_open=True), brain)
        collect(guard)
        await guard.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.6), DOWN)
        return brain.calls
    assert asyncio.run(go()) == ["turn_ended"]


# ------------------------------------------------------------------------------------------------ the brain
class Turn:
    def __init__(self):
        self.debug = type("D", (), {"intent": "none", "via": "rules", "draft": False, "risk": "low", "note": ""})()
        self.ended = False


class Session:
    def __init__(self):
        self.history = []

    def remember(self, who, text):
        self.history.append((who, text))


class Engine:
    """Answers with one sentence, remembers the turn as the real engine does when it finishes; optionally waits for `release` first."""

    def __init__(self, hold=None):
        self.seen, self.hold, self.started = [], hold, threading.Event()

    def new_session(self):
        return Session()

    def handle_stream(self, s, text, **extra):
        self.seen.append(text)
        self.started.set()
        if self.hold is not None:
            self.hold.wait(5)
        yield f"Reply to: {text}"
        s.remember("customer", text)
        s.remember("agent", f"Reply to: {text}")
        yield Turn()


def make(engine):
    heard, pushed = [], []
    brain = DealerBrain(engine, speak_opening=False, on_heard=heard.append)

    async def push(frame, direction=DOWN):
        pushed.append(frame)
    brain.push_frame = push
    return brain, heard, pushed


def transcript(text):
    return TranscriptionFrame(text, "user", "2026-10-27T22:00:00Z")


def test_a_take_back_before_the_transcript_arrives_joins_the_two_halves():
    async def go():
        eng = Engine()
        brain, heard, pushed = make(eng)
        assert await brain.take_back() is False                                   # nothing to throw away yet: wait for the first half's words
        await brain.process_frame(transcript("I'm looking for a"), DOWN)
        assert eng.seen == [] and brain.continued == 1                            # kept, not answered
        await brain.process_frame(transcript("Toyota"), DOWN)
        return eng.seen, heard
    seen, heard = asyncio.run(go())
    assert seen == ["I'm looking for a Toyota"] and heard == ["I'm looking for a Toyota"]


def test_a_take_back_while_the_first_half_is_being_answered_drops_that_answer_and_joins():
    async def go():
        release = threading.Event()
        eng = Engine(hold=release)
        brain, heard, pushed = make(eng)
        task = asyncio.create_task(brain._answer("I'm looking for a"))
        await asyncio.get_running_loop().run_in_executor(None, eng.started.wait, 5)
        assert await brain.take_back() is True                                    # the guard now interrupts the pipeline
        task.cancel()                                                             # (what the interruption does to the turn being processed)
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()                                                             # only now may the engine's thread run on: it finds the turn cancelled and stops
        await asyncio.sleep(0.2)
        assert brain.session.history == [] and heard == []                        # nothing remembered, nothing shown
        eng.hold = None
        await brain._answer("Toyota")
        return eng.seen, heard, brain.session.history
    seen, heard, history = asyncio.run(go())
    assert seen == ["I'm looking for a", "I'm looking for a Toyota"] and heard == ["I'm looking for a Toyota"]
    assert history == [("customer", "I'm looking for a Toyota"), ("agent", "Reply to: I'm looking for a Toyota")]


def test_a_take_back_after_the_reply_was_written_but_before_it_is_heard_forgets_the_exchange():
    async def go():
        eng = Engine()
        brain, heard, pushed = make(eng)
        await brain._answer("I'm looking for a")
        assert brain.session.history and brain._open_turn == "I'm looking for a"
        assert await brain.take_back() is True                                    # the reply is queued for the voice: it must be thrown away
        assert brain.session.history == []
        await brain._answer("Toyota")
        return eng.seen, brain.session.history
    seen, history = asyncio.run(go())
    assert seen == ["I'm looking for a", "I'm looking for a Toyota"] and history == [("customer", "I'm looking for a Toyota"), ("agent", "Reply to: I'm looking for a Toyota")]


def test_once_the_reply_is_heard_it_cannot_be_taken_back():
    async def go():
        brain, _, _ = make(Engine())
        await brain._answer("what colours do you have")
        await brain.process_frame(BotStartedSpeakingFrame(), DOWN)
        return brain._open_turn
    assert asyncio.run(go()) is None


def test_a_stale_wait_does_not_swallow_the_next_real_turn():
    async def go():
        eng = Engine()
        brain, heard, _ = make(eng)
        await brain.take_back()                                                   # the first half never produced a transcript...
        brain.turn_ended()                                                        # ...and the next turn ended: forget the wait
        await brain.process_frame(transcript("what colours do you have"), DOWN)
        return eng.seen
    assert asyncio.run(go()) == ["what colours do you have"]


def test_a_turn_with_nothing_to_say_still_shows_what_the_caller_said():
    class Silent(Engine):
        def handle_stream(self, s, text, **extra):
            yield Turn()
    async def go():
        brain, heard, _ = make(Silent())
        await brain._answer("hmm")
        return heard
    assert asyncio.run(go()) == ["hmm"]


def test_an_ordinary_barge_in_is_still_remembered_as_interrupted():
    async def go():
        release = threading.Event()
        eng = Engine(hold=release)
        brain, heard, _ = make(eng)
        task = asyncio.create_task(brain._answer("tell me about the RAV4"))
        await asyncio.get_running_loop().run_in_executor(None, eng.started.wait, 5)
        task.cancel()                                                             # no take_back: the caller spoke over a reply they were hearing
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await asyncio.sleep(0.2)
        return brain.session.history, brain.interrupted
    history, interrupted = asyncio.run(go())
    assert history == [("customer", "tell me about the RAV4"), ("agent", "(interrupted)")] and interrupted == 1


def test_the_setting_and_the_wiring():
    import inspect

    from dealer_core.config import load_config
    from dealer_core.server.app import run_call
    from dealer_core.voice import pipeline

    assert load_config(environ={}).voice.tap_grace_secs == 2.0
    assert load_config(environ={"VA__VOICE__TAP_GRACE_SECS": "0"}).voice.tap_grace_secs == 0
    assert "ContinuationGuard(mic_gate, brain)" in inspect.getsource(pipeline.build_pipeline)
    src = inspect.getsource(run_call)
    assert "hold_open_until_reply" in src and '"thinking"' in src and "gate.close_after_turn()" in src      # the old behaviour is still there for grace 0
