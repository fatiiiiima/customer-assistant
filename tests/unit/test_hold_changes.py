"""The shorter hold (ARCH_REVIEW section 92): the reply written while Smart Turn holds a pause (voice.hold_reply), the turn ended when the words read as finished
(voice.smart_text_check), and the shorter settings in one switch (voice.fast_hold)."""
import asyncio
import concurrent.futures
import threading
import time

import pytest
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.frames.frames import InputAudioRawFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

from dealer_core.voice.endpoint import SmartEndpointer
from dealer_core.voice.finished import reads_finished
from dealer_core.voice.speculate import Speculation

from .test_endpoint import FakeAnalyzer
from .test_model_only import SAYS, Model, Parallel, model_only


# ------------------------------------------------------------------------------------------------ do the words read as finished?
@pytest.mark.parametrize("text", [
    "I'm looking for Hyundai.",                       # the 17:35 call, held 0.85 s
    "This Santa Fe.",
    "What color is it?",                              # a question ends on a pronoun and is whole
    "Did it have a previous owner?",
    "How many kilometers?",
    "Tell me about it.",
    "I can only do a hundred thousand demos.",
])
def test_words_that_end_like_a_sentence_read_as_finished(text):
    assert reads_finished(text)


@pytest.mark.parametrize("text", [
    "I'm looking for a",                              # no end mark, ends on an article
    "I'm looking for a Toyota,",                      # a comma is a pause
    "I'm looking for the...",
    "I want the cheaper one and",
    "I'm looking for a Toyota with.",                 # a mark, but it ends on a word that needs another
    "What's the price of the.",
    "Yes.",                                           # one word: the answer check knows the question, this does not
    "What?",
    "Um.",
    "Uh, um.",
    "",
    "   ",
])
def test_words_that_are_not_over_do_not(text):
    assert not reads_finished(text)


# ------------------------------------------------------------------------------------------------ the speculation writes the reply and hands over known words
class Handle:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


def make(prepare, decide=lambda t: {"tool": "search_cars"}, version=None, transcribe=lambda pcm: "anything for the desert"):
    return Speculation(transcribe, decide, version or (lambda: 3), prepare)


def test_the_reply_is_written_from_the_words_and_the_decision_and_used_when_the_turn_ends():
    seen, handle = [], Handle()

    def prepare(text, decision):
        seen.append((text, decision))
        return handle

    async def go():
        spec = make(prepare)
        spec.start(b"\1\0" * 8000)
        spec.confirm()
        text = await spec.transcript()
        reply = spec.reply_for(text)
        return reply.result(timeout=2), spec.reply_used

    got, used = asyncio.run(go())
    assert got is handle and used == 1 and not handle.cancelled
    assert seen == [("anything for the desert", {"tool": "search_cars"})]


def test_the_reply_is_ready_before_the_decision_is_handed_on():
    """The turn takes the decision first and then asks for the reply: it must already be there."""
    order = []

    async def go():
        spec = make(lambda t, d: order.append("reply") or Handle())
        spec.start(b"\1\0" * 8000)
        spec.confirm()
        text = await spec.transcript()
        d = spec.decision_for(text)
        d.result(timeout=2)
        done_after_decision = spec.reply_for(text).done()
        return done_after_decision

    assert asyncio.run(go()) is True


def test_a_reply_written_for_a_hold_the_caller_ended_by_speaking_again_is_stopped():
    handle, made = Handle(), threading.Event()

    def prepare(text, decision):
        made.set()
        return handle

    async def go():
        spec = make(prepare)
        spec.start(b"\1\0" * 8000)
        for _ in range(200):
            await asyncio.sleep(0.01)
            if made.is_set() and spec._reply.done():
                break
        spec.resumed()                                 # the caller carried on
        spec.confirm()
        return spec.reply_for("anything for the desert")

    assert asyncio.run(go()) is None and handle.cancelled


def test_a_reply_that_is_ready_after_the_caller_spoke_again_is_stopped_at_once():
    handle, release = Handle(), threading.Event()

    def prepare(text, decision):
        release.wait(2)                                # still being made when the caller speaks
        return handle

    async def go():
        spec = make(prepare)
        spec.start(b"\1\0" * 8000)
        await asyncio.sleep(0.05)
        spec.resumed()
        release.set()
        await asyncio.sleep(0.1)

    asyncio.run(go())
    assert handle.cancelled


def test_the_reply_is_not_used_when_the_words_differ_or_the_call_moved_on_and_is_then_stopped():
    for case in ("other words", "moved on"):
        handle, version = Handle(), {"v": 3}

        async def go():
            spec = make(lambda t, d: handle, version=lambda: version["v"])
            spec.start(b"\1\0" * 8000)
            spec.confirm()
            text = await spec.transcript()
            await asyncio.wrap_future(spec._reply)
            if case == "moved on":
                version["v"] = 4
            return spec.reply_for("a black one" if case == "other words" else text)

        assert asyncio.run(go()) is None, case
        assert handle.cancelled, case


def test_no_reply_is_made_without_a_prepare_hook_and_a_hook_that_fails_is_just_no_reply():
    async def go(prepare):
        spec = make(prepare)
        spec.start(b"\1\0" * 8000)
        spec.confirm()
        text = await spec.transcript()
        r = spec.reply_for(text)
        return None if r is None else r.result(timeout=2), spec.decision_for(text)

    assert asyncio.run(go(None))[0] is None                                    # no hook: nothing to hand over
    boom = lambda t, d: (_ for _ in ()).throw(RuntimeError("no"))
    reply, decision = asyncio.run(go(boom))
    assert reply is None and decision is not None                              # the decision still arrives


def test_words_that_are_already_known_are_not_recognised_again():
    def never(pcm):
        raise AssertionError("recognised a second time")

    async def go():
        spec = make(lambda t, d: Handle(), transcribe=never)
        spec.start(b"\1\0" * 8000, "I'm looking for Hyundai.")
        spec.confirm()
        text = await asyncio.wait_for(spec.transcript(), 1)
        return text, spec.decision_for(text).result(timeout=2)

    text, decision = asyncio.run(go())
    assert text == "I'm looking for Hyundai." and decision == {"tool": "search_cars"}


# ------------------------------------------------------------------------------------------------ the endpointer
class Spy:
    """Stands in for the speculation: records what the endpointer does to it."""

    def __init__(self):
        self.calls = []

    def start(self, pcm, text=None):
        self.calls.append(("start", len(pcm) > 0, text))

    def resumed(self):
        self.calls.append(("resumed",))

    def confirm(self):
        self.calls.append(("confirm",))

    def clear(self):
        self.calls.append(("clear",))


def loud(secs=0.5):
    return InputAudioRawFrame(audio=b"\1\0" * int(16000 * secs), sample_rate=16000, num_channels=1)


def drive(verdicts, heard, finished=reads_finished, answer=None, frames=None):
    ep = SmartEndpointer(analyzer=FakeAnalyzer(verdicts, fallback_after=None))
    ep.speculation, out, notes = Spy(), [], []

    async def peek(pcm):
        return heard

    ep.peek, ep.reads_finished, ep.is_answer = peek, finished, answer
    ep.on_note = lambda kind, text: notes.append((kind, text))

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        out.append(type(frame).__name__)
    ep.push_frame = push

    async def go():
        for f in frames or [VADUserStartedSpeakingFrame(start_secs=0.1), loud(), VADUserStoppedSpeakingFrame(stop_secs=0.15)]:
            await ep.process_frame(f, FrameDirection.DOWNSTREAM)
    asyncio.run(go())
    return ep, out, notes


def test_a_turn_smart_turn_holds_ends_at_once_when_the_words_read_as_finished_and_they_are_handed_over():
    ep, out, notes = drive([EndOfTurnState.INCOMPLETE], "I'm looking for Hyundai.")
    assert "VADUserStoppedSpeakingFrame" in out and ep.held == 0 and ep.text_finished == 1
    assert ep.speculation.calls == [("clear",), ("start", True, "I'm looking for Hyundai."), ("confirm",)]       # one start, from the known words, and it applies
    assert notes[0][0] == "text_check" and "finished sentence" in notes[0][1]


def test_a_turn_whose_words_are_not_over_stays_held_and_the_speculation_starts_from_the_same_words_once():
    ep, out, notes = drive([EndOfTurnState.INCOMPLETE], "I'm looking for a")
    assert "VADUserStoppedSpeakingFrame" not in out and ep.held == 1 and ep.text_finished == 0
    assert ep.speculation.calls == [("clear",), ("start", True, "I'm looking for a")]       # not recognised again for the hold, and no confirm yet
    assert "stays" in notes[0][1]


def test_when_smart_turn_says_complete_the_words_are_not_even_recognised():
    ep, out, _ = drive([EndOfTurnState.COMPLETE], "never asked")
    assert ep.peeked == 0 and "VADUserStoppedSpeakingFrame" in out


def test_without_the_hook_the_hold_is_as_before():
    ep = SmartEndpointer(analyzer=FakeAnalyzer([EndOfTurnState.INCOMPLETE]))
    ep.speculation, out = Spy(), []

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        out.append(type(frame).__name__)
    ep.push_frame = push

    async def go():
        for f in (VADUserStartedSpeakingFrame(start_secs=0.1), loud(), VADUserStoppedSpeakingFrame(stop_secs=0.15)):
            await ep.process_frame(f, FrameDirection.DOWNSTREAM)
    asyncio.run(go())
    assert ep.held == 1 and ep.speculation.calls == [("clear",), ("start", True, None)] and ep.peeked == 0


def test_an_answer_to_the_agents_question_still_ends_the_turn_and_is_counted_apart():
    ep, out, notes = drive([EndOfTurnState.INCOMPLETE], "Toyota", answer=lambda t: True)
    assert "VADUserStoppedSpeakingFrame" in out and ep.short_answers == 1 and ep.text_finished == 0
    assert "complete answer" in notes[0][1]


def test_a_peek_that_fails_keeps_the_hold():
    async def peek(pcm):
        raise RuntimeError("server down")

    ep = SmartEndpointer(analyzer=FakeAnalyzer([EndOfTurnState.INCOMPLETE]))
    ep.speculation, ep.peek, ep.reads_finished, out = Spy(), peek, reads_finished, []

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        out.append(type(frame).__name__)
    ep.push_frame = push

    async def go():
        for f in (VADUserStartedSpeakingFrame(start_secs=0.1), loud(), VADUserStoppedSpeakingFrame(stop_secs=0.15)):
            await ep.process_frame(f, FrameDirection.DOWNSTREAM)
    asyncio.run(go())
    assert ep.held == 1 and "VADUserStoppedSpeakingFrame" not in out and ep.speculation.calls[-1] == ("start", True, None)      # held, and the speculation recognises as before


# ------------------------------------------------------------------------------------------------ the engine writes the reply during the hold
def hold_engine(decisions, **kw):
    b = Parallel(decisions, **kw)
    from .test_scale import agent
    e, _ = agent(None, backend=b)
    e.model_only = e.reply_writer.model_only = e.hold_reply = True
    e.silent_fallback = False
    return e, b


def done(value):
    f = concurrent.futures.Future()
    f.set_result(value)
    return f


def test_hold_the_reply_written_during_the_hold_is_the_one_spoken_with_no_second_writing_call():
    e, b = hold_engine(["search_cars"])
    s, text = e.new_session(), "do you have any Toyotas"
    decision = e.predecide(s, text)                                 # made during the hold
    spec = e.prepare_reply(s, text, decision)
    assert spec is not None and spec.key == "search_cars"
    out = list(e.handle_stream(s, text, predecided=done(decision), reply_ahead=done(spec)))
    r = out[-1]
    assert r.reply == "Here is what I have. Want to hear more?" and r.debug.intent == "search_cars" and r.debug.reply_ahead is True
    assert "written during the hold" in r.debug.note and len(b.writes) == 1 and len(b.prompts) == 1      # one decision, ONE reply, written once
    assert s.shown and s.history[-1] == ("agent", r.reply)           # the call carries on from the real lookup, not from the copy


def test_hold_the_parallel_replies_are_not_written_as_well():
    e, b = hold_engine(["search_cars"])
    e.parallel = True
    s, text = e.new_session(), "do you have any Toyotas"
    decision = e.predecide(s, text)
    spec = e.prepare_reply(s, text, decision)
    list(e.handle_stream(s, text, predecided=done(decision), reply_ahead=done(spec)))
    assert len(b.writes) == 1                                       # a decision made ahead needs no guesses at what it will be


def test_hold_a_reply_for_another_lookup_than_the_decision_is_stopped_and_the_right_one_written_now():
    e, b = hold_engine(["search_cars"])
    s, text = e.new_session(), "do you have any Toyotas"
    wrong = e.prepare_reply(s, text, {"tool": "none"})              # the decision made during the hold said "none" ...
    assert wrong.key == "none"
    out = list(e.handle_stream(s, text, predecided=done({"tool": "search_cars"}), reply_ahead=done(wrong)))   # ...and the turn's own decision is a search
    r = out[-1]
    assert r.debug.reply_ahead is False and "not search_cars" in r.debug.note and r.debug.intent == "search_cars"
    assert wrong.stop.is_set()
    assert r.reply == "Here is what I have. Want to hear more?"


def test_hold_a_changed_prompt_is_written_again():
    e, b = hold_engine(["search_cars"])
    s, text = e.new_session(), "do you have any Toyotas"
    spec = e.prepare_reply(s, text, {"tool": "search_cars"})
    s.remember("customer", "hello")                                 # the call moved on after the reply was written
    s.remember("agent", "Hi.")
    r = list(e.handle_stream(s, text, predecided=done({"tool": "search_cars"}), reply_ahead=done(spec)))[-1]
    assert "had changed" in r.debug.note and r.reply == "Here is what I have. Want to hear more?"
    assert len(b.writes) == 2                                       # the one written ahead (not spoken) and the one written now


@pytest.mark.parametrize("tool", ["handoff", "confirm_booking", "cancel_booking", "reschedule_booking", "compare_cars"])
def test_hold_nothing_that_acts_outside_the_call_is_ever_written_ahead(tool):
    e, b = hold_engine([tool])
    s = e.new_session()
    assert e.prepare_reply(s, "whatever", {"tool": tool}) is None and not b.writes


def test_hold_a_failed_or_empty_decision_writes_nothing():
    e, b = hold_engine([])
    s = e.new_session()
    assert e.prepare_reply(s, "x", {"error": "timeout"}) is None
    assert e.prepare_reply(s, "x", None) is None                    # no decision and no rules_decide: nothing to write
    assert not b.writes


def test_hold_with_rules_decide_the_code_decided_turn_gets_its_reply_written_during_the_hold():
    b = Model([])
    from .test_scale import agent
    e, _ = agent(None, backend=b)
    e.model_only = e.reply_writer.model_only = e.hold_reply = e.rules_decide = True
    e.silent_fallback = False
    s, text = e.new_session(), "tell me about the Land Cruiser"
    spec = e.prepare_reply(s, text, None)
    assert spec is not None and spec.key.startswith("get_car ")
    r = list(e.handle_stream(s, text, reply_ahead=done(spec)))[-1]
    assert r.reply == SAYS and not b.prompts and r.debug.reply_ahead is True and len(b.writes) == 1


def test_hold_it_is_faster_than_deciding_then_writing_by_about_a_whole_call():
    """The decision takes 0.3 s and the reply 0.3 s. During the hold both run; the turn then only collects them."""
    e, _ = hold_engine(["search_cars"], decide_s=0.3, write_s=0.3)
    s, text = e.new_session(), "do you have any Toyotas"
    decision = e.predecide(s, text)
    spec = e.prepare_reply(s, text, decision)
    time.sleep(0.35)                                                # the reply is written while the hold runs out
    t = time.perf_counter()
    out = list(e.handle_stream(s, text, predecided=done(decision), reply_ahead=done(spec)))
    assert out[-1].reply and time.perf_counter() - t < 0.2, time.perf_counter() - t


def test_hold_is_off_without_the_test_mode_and_by_default():
    from dealer_core.app import build_app
    from dealer_core.config import load_config
    from dealer_core.voice.pipeline import apply_voice_profile
    from .test_scale import Scripted

    assert load_config(environ={}).voice.hold_reply is False
    for mo, hold, want in (("false", "true", False), ("true", "true", True), ("true", "false", False)):
        cfg = load_config(environ={"VA__VOICE__MODEL_ONLY": mo, "VA__VOICE__HOLD_REPLY": hold})
        b = Scripted()
        app = build_app(cfg, classifier=b, reply_mode="conversational", lead_backend=b, engine="agent")
        apply_voice_profile(app, cfg)
        assert app.engine.hold_reply is want, (mo, hold)
    e, b = model_only(["search_cars"])
    assert e.hold_reply is False and e.prepare_reply(e.new_session(), "x", {"tool": "none"}) is None       # the test mode alone writes nothing ahead


# ------------------------------------------------------------------------------------------------ the shorter settings
def test_fast_hold_is_off_by_default_so_nothing_changes():
    from dealer_core.config import load_config

    v = load_config(environ={}).voice
    assert (v.fast_hold, v.smart_text_check, v.smart_fallback_secs, v.smart_stop_secs) == (False, False, 0.6, 0.2)


def test_fast_hold_shortens_the_fallback_and_the_detectors_silence():
    from dealer_core.config import load_config

    v = load_config(environ={"VA__VOICE__FAST_HOLD": "true"}).voice
    assert (v.smart_fallback_secs, v.smart_stop_secs) == (0.4, 0.15)


def test_fast_hold_only_ever_shortens():
    from dealer_core.config import load_config

    v = load_config(environ={"VA__VOICE__FAST_HOLD": "true", "VA__VOICE__SMART_FALLBACK_SECS": "0.3", "VA__VOICE__SMART_STOP_SECS": "0.1"}).voice
    assert (v.smart_fallback_secs, v.smart_stop_secs) == (0.3, 0.1)


def test_the_text_check_reaches_the_endpointer_only_when_it_is_on():
    """The pipeline gives the endpointer the check, and a peek that cleans the words the way the speculation does."""
    import inspect

    from dealer_core.voice import pipeline

    src = inspect.getsource(pipeline.build_pipeline)
    assert "vcfg.smart_text_check" in src and "endpointer.reads_finished = reads_finished" in src and "prepare_reply" in src


def test_in_the_test_mode_the_model_decides_even_the_turns_the_rules_could_route_so_the_hold_makes_that_decision_too():
    b2 = Parallel(["get_car"])
    from .test_scale import agent
    e, _ = agent(None, backend=b2)
    e.model_only = e.reply_writer.model_only = True
    assert e.predecide(e.new_session(), "tell me about the Land Cruiser") == {"tool": "get_car", "car_id": None} and len(b2.prompts) == 1
    e.rules_decide = True                                           # with rules_decide the code takes it: nothing to ask the model
    assert e.predecide(e.new_session(), "tell me about the Land Cruiser") is None and len(b2.prompts) == 1
