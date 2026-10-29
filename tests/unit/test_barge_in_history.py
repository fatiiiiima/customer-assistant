"""After a barge-in the call keeps only what the caller heard of the agent's reply (owner, 29 Oct: "while the model is speaking, if I interrupt it, it is able to
understand and changes the response"). The reply is saved when it is written, about a second in, and spoken for several seconds more: an interruption used to leave
all of it in the history, as if the caller had heard it."""
import asyncio
import sys
import threading
from pathlib import Path

import pytest

from dealer_core.voice.brain import DealerBrain

sys.path.insert(0, str(Path(__file__).parent))

LIST = "There's the Corolla at 62,500 dirhams, the Camry at 74,500, and the RAV4 at 98,500."
QUESTION = "Which one would you like to hear more about?"


class Car:
    def __init__(self, model):
        self.model = model


class Session:
    def __init__(self):
        self.history, self.shown = [], [Car("Corolla"), Car("Camry"), Car("RAV4")]

    def remember(self, who, text):
        self.history.append((who, text))


class Turn:
    def __init__(self):
        self.debug = type("D", (), {"intent": "search_cars", "via": "rules", "draft": False, "risk": "low", "note": ""})()
        self.ended = False


class Engine:
    def __init__(self, hold=None):
        self.hold, self.started = hold, threading.Event()

    def new_session(self):
        return Session()

    def handle_stream(self, s, text, **extra):
        self.started.set()
        yield LIST
        if self.hold is not None:
            self.hold.wait(5)
        yield QUESTION
        s.remember("customer", text)
        s.remember("agent", f"{LIST} {QUESTION}")
        yield Turn()


def brain_for(engine):
    b = DealerBrain(engine, speak_opening=False)

    async def push(frame, direction=None):
        pass
    b.push_frame = push
    return b


def test_a_reply_cut_after_it_was_written_keeps_only_what_was_heard():
    b = brain_for(Engine())
    asyncio.run(b._answer("Any Toyotas?"))
    b.reply_cut("There's the Corolla at 62,500 dirhams, the Camry ...")
    assert b.session.history[-1] == ("agent", "There's the Corolla at 62,500 dirhams, the Camry ... (interrupted)")
    assert [c.model for c in b.session.shown] == ["Corolla", "Camry"]          # "the last one" now means the Camry, the last one the caller heard
    assert b.interrupted == 1


def test_a_reply_cut_while_it_was_still_being_written_keeps_only_what_was_heard():
    async def go():
        release = threading.Event()
        eng = Engine(hold=release)
        b = brain_for(eng)
        task = asyncio.create_task(b._answer("Any Toyotas?"))
        await asyncio.get_running_loop().run_in_executor(None, eng.started.wait, 5)
        await asyncio.sleep(0.05)
        b.reply_cut("There's the Corolla ...")
        task.cancel()                                                         # what the interruption does to the turn being processed
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        return b
    b = asyncio.run(go())
    assert b.session.history == [("customer", "Any Toyotas?"), ("agent", "There's the Corolla ... (interrupted)")]
    assert [c.model for c in b.session.shown] == ["Corolla"]


def test_a_cut_that_is_not_the_saved_reply_changes_nothing():
    """A "speak up" request is spoken but not saved: cutting it must not overwrite the reply before it."""
    b = brain_for(Engine())
    asyncio.run(b._answer("Any Toyotas?"))
    before = list(b.session.history)
    b.reply_cut("Sorry, could you say that ...")
    assert b.session.history == before and b.interrupted == 0


def test_a_cut_before_any_word_was_heard_changes_nothing():
    b = brain_for(Engine())
    asyncio.run(b._answer("Any Toyotas?"))
    before = list(b.session.history)
    b.reply_cut("")
    assert b.session.history == before


def test_the_next_answer_follows_what_was_heard():
    """The real agent: the Toyota list is cut after the Camry, the caller says "the last one", and gets the Camry, not a car they never heard."""
    import test_sweep_fixes as F

    e, _ = F.at("search")
    b = DealerBrain(e, speak_opening=False)

    async def push(frame, direction=None):
        pass
    b.push_frame = push

    async def go():
        for line in ("I'm looking for a Toyota.", "Any model and any budget is fine."):
            await b._answer(line)
        listed = b.session.history[-1][1]
        assert "Land Cruiser" in listed or "RAV4" in listed
        heard = listed[: listed.index("Camry") + len("Camry")] + " ..."
        b.reply_cut(heard)
        await b._answer("Tell me about the last one.")
    asyncio.run(go())
    assert b.session.history[-3][1].endswith("(interrupted)")
    assert "Camry" in b.session.history[-1][1] and e.svc.get_car(b.session.car).model == "Camry"


class Echo:
    """Remembers every text it was asked to answer."""

    def __init__(self):
        self.seen = []

    def new_session(self):
        return Session()

    def handle_stream(self, s, text, **extra):
        self.seen.append(text)
        yield f"Reply to: {text}"
        yield Turn()


def test_a_turn_that_is_only_actually_waits_for_the_rest():
    """29 Oct, a barge-in: "Actually," then a pause, then "do you have any BMWs?" were answered as two turns."""
    from pipecat.frames.frames import TranscriptionFrame

    async def go():
        eng = Echo()
        b = brain_for(eng)
        b.marker_wait_s = 0.5
        await b.process_frame(TranscriptionFrame("Actually.", "user", "t"), None)
        await asyncio.sleep(0.1)
        assert eng.seen == []                                                 # held, not answered
        await b.process_frame(TranscriptionFrame("Do you have any BMWs?", "user", "t"), None)
        await asyncio.sleep(0.6)
        return eng.seen
    assert asyncio.run(go()) == ["Actually. Do you have any BMWs?"]


def test_a_held_actually_is_answered_alone_if_nothing_follows_and_yes_is_never_held():
    from pipecat.frames.frames import TranscriptionFrame

    async def go():
        eng = Echo()
        b = brain_for(eng)
        b.marker_wait_s = 0.2
        await b.process_frame(TranscriptionFrame("Wait.", "user", "t"), None)
        await asyncio.sleep(0.4)
        await b.process_frame(TranscriptionFrame("Yes.", "user", "t"), None)
        await b.process_frame(TranscriptionFrame("No.", "user", "t"), None)
        return eng.seen
    assert asyncio.run(go()) == ["Wait.", "Yes.", "No."]


def test_while_the_caller_goes_on_speaking_after_actually_the_hold_lasts():
    """The rest of the sentence takes longer to say and recognise than the hold: speech after "Actually," keeps it waiting for that transcript."""
    from pipecat.frames.frames import TranscriptionFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame

    async def go():
        eng = Echo()
        b = brain_for(eng)
        b.marker_wait_s = 0.2
        await b.process_frame(TranscriptionFrame("Actually,", "user", "t"), None)
        await b.process_frame(VADUserStartedSpeakingFrame(start_secs=0.1), None)
        await asyncio.sleep(0.5)                                              # longer than the hold: the caller is still talking
        assert eng.seen == []
        await b.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.3), None)
        await b.process_frame(TranscriptionFrame("do you have any Porsches?", "user", "t"), None)
        await asyncio.sleep(0.1)
        return eng.seen
    assert asyncio.run(go()) == ["Actually, do you have any Porsches?"]


def test_when_the_rest_starts_before_the_words_of_actually_arrive_it_still_waits():
    """The synthetic barge-in call: the caller carried on 0.1 s after "Actually,", and its words arrived from the recogniser after that."""
    from pipecat.frames.frames import TranscriptionFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame

    async def go():
        eng = Echo()
        b = brain_for(eng)
        b.marker_wait_s = 0.2
        await b.process_frame(VADUserStartedSpeakingFrame(start_secs=0.1), None)
        await b.process_frame(TranscriptionFrame("Actually.", "user", "t"), None)
        await asyncio.sleep(0.5)
        assert eng.seen == []
        await b.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.3), None)
        await b.process_frame(TranscriptionFrame("Do you have any Porsches?", "user", "t"), None)
        await asyncio.sleep(0.1)
        return eng.seen
    assert asyncio.run(go()) == ["Actually. Do you have any Porsches?"]
