"""Two ways to cut the wait for the first words (ARCH_REVIEW section 95): the work starts after a short quiet inside every turn (voice.early_start_secs, now gated and
verified), and a short acknowledgement the code picks is said while the model writes (voice.opener_first, with model_only)."""
import asyncio
import threading
import time

import numpy as np
import pytest
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.frames.frames import InputAudioRawFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

from dealer_core.agent.wording import NEUTRAL_OPENERS, OPENERS
from dealer_core.voice.endpoint import SmartEndpointer
from dealer_core.voice.finished import reads_finished
from dealer_core.voice.speculate import Speculation, same_words

from .test_endpoint import FakeAnalyzer
from .test_model_only import Model, Parallel
from .test_scale import agent

FRAME = 320


def loud(amplitude=8000):
    return InputAudioRawFrame(audio=(np.sin(np.arange(FRAME) / 3) * amplitude).astype("<i2").tobytes(), sample_rate=16000, num_channels=1)


def quiet():
    return InputAudioRawFrame(audio=(np.random.default_rng(1).normal(0, 20, FRAME)).astype("<i2").tobytes(), sample_rate=16000, num_channels=1)


class Handle:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


def wait_for(cond, secs=2.0):
    end = time.time() + secs
    while time.time() < end and not cond():
        time.sleep(0.01)
    return cond()


# ------------------------------------------------------------------------------------------------ the early start, gated
def spec_with(text, gate=None, verify=False):
    calls = {"decide": 0, "prepare": 0}

    def decide(t):
        calls["decide"] += 1
        return {"tool": "search_cars"}

    def prepare(t, d):
        calls["prepare"] += 1
        return Handle()

    sp = Speculation(lambda pcm: text, decide, lambda: 3, prepare)
    sp.gate, sp.verify_early = gate, verify
    return sp, calls


def test_a_half_sentence_is_recognised_and_nothing_more_is_spent_on_it():
    sp, calls = spec_with("Actually, I'm looking for the", gate=reads_finished)
    events = []
    sp.on_event = lambda k, t: events.append(k)

    async def go():
        sp.start_early(b"\1\0" * 8000)
        sp.confirm()
        text = await sp.transcript()
        return text, sp.decision_for("Actually, I'm looking for the"), sp.reply_for("Actually, I'm looking for the")

    asyncio.run(go())
    assert wait_for(lambda: "early_skipped" in events) and calls == {"decide": 0, "prepare": 0} and sp.skipped == 1


def test_a_finished_sentence_gets_its_decision_and_its_reply_started():
    sp, calls = spec_with("I'm looking for Hyundai.", gate=reads_finished)

    async def go():
        sp.start_early(b"\1\0" * 8000)
        for _ in range(100):
            await asyncio.sleep(0.01)
            if sp._decision.done():
                break
        return sp.decision_for("I'm looking for Hyundai.").result(timeout=1)

    assert asyncio.run(go()) == {"tool": "search_cars"} and calls == {"decide": 1, "prepare": 1}


def test_the_gate_applies_to_early_starts_only_a_held_pause_still_gets_its_work_whatever_it_says():
    sp, calls = spec_with("Actually, I'm looking for the", gate=reads_finished)

    async def go():
        sp.start(b"\1\0" * 8000)                                       # Smart Turn held a pause
        for _ in range(100):
            await asyncio.sleep(0.01)
            if sp._decision.done():
                break

    asyncio.run(go())
    assert calls == {"decide": 1, "prepare": 1}


def test_with_the_check_on_the_early_words_are_not_handed_over_but_the_early_work_is_kept_for_the_same_words():
    sp, calls = spec_with("Do you have a Santa Fe?", verify=True)
    handle = []

    async def go():
        sp.start_early(b"\1\0" * 8000)
        sp.confirm()
        words = await sp.transcript()                                  # None: the service recognises the turn in full, as usual
        for _ in range(100):
            await asyncio.sleep(0.01)
            if sp._reply.done():
                break
        d = sp.decision_for("do you have a Santa Fe")                  # the full reading, without the question mark
        r = sp.reply_for("do you have a Santa Fe")
        return words, d, r

    words, d, r = asyncio.run(go())
    assert words is None and d is not None and r is not None and sp.reply_used == 1


def test_with_the_check_on_a_different_reading_throws_the_early_work_away():
    sp, calls = spec_with("Do you have a Santa Fe?", verify=True)

    async def go():
        sp.start_early(b"\1\0" * 8000)
        sp.confirm()
        await sp.transcript()
        for _ in range(100):
            await asyncio.sleep(0.01)
            if sp._reply.done():
                break
        handle = sp._reply.result()
        return sp.decision_for("do you have a Sante"), sp.reply_for("do you have a Sante"), handle

    d, r, handle = asyncio.run(go())
    assert d is None and r is None and handle.cancelled


def test_a_held_pause_still_hands_over_its_words_with_the_check_on():
    sp, _ = spec_with("a white one", verify=True)

    async def go():
        sp.start(b"\1\0" * 8000)
        sp.confirm()
        return await sp.transcript()

    assert asyncio.run(go()) == "a white one"


def test_same_words_ignores_punctuation_and_case():
    assert same_words("Do you have a Santa Fe?", "do you have a santa fe") and same_words("It's 9.", "it's 9")
    assert not same_words("a white one", "a black one") and not same_words("yes", "")


def test_the_endpointer_starts_the_gated_work_after_the_quiet_inside_a_turn_and_confirms_it_at_the_end():
    sp, calls = spec_with("I'm looking for Hyundai.", gate=reads_finished, verify=True)
    ep = SmartEndpointer(analyzer=FakeAnalyzer([EndOfTurnState.COMPLETE]), early_start_secs=0.12)
    ep.speculation = sp
    out = []

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        out.append(type(frame).__name__)
    ep.push_frame = push

    async def go():
        for f in [VADUserStartedSpeakingFrame(start_secs=0.1), *[loud() for _ in range(10)], *[quiet() for _ in range(7)]]:      # 0.14 s of quiet
            await ep.process_frame(f, FrameDirection.DOWNSTREAM)
        for _ in range(100):
            await asyncio.sleep(0.01)
            if sp._reply is not None and sp._reply.done():
                break
        await ep.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2), FrameDirection.DOWNSTREAM)       # the turn ends
        return sp.decision_for("I'm looking for Hyundai").result(timeout=1), sp.reply_for("I'm looking for Hyundai")

    d, r = asyncio.run(go())
    assert ep.early_starts == 1 and d == {"tool": "search_cars"} and r is not None and calls == {"decide": 1, "prepare": 1}
    assert "VADUserStoppedSpeakingFrame" in out


def test_the_settings_and_their_wiring():
    import inspect

    from dealer_core.config import load_config
    from dealer_core.voice import pipeline

    v = load_config(environ={}).voice
    assert (v.early_start_secs, v.early_start_gate, v.early_start_verify, v.opener_first) == (0.0, True, True, False)
    v = load_config(environ={"VA__VOICE__EARLY_START_SECS": "0.1", "VA__VOICE__OPENER_FIRST": "true"}).voice
    assert v.early_start_secs == 0.1 and v.opener_first is True
    src = inspect.getsource(pipeline.build_pipeline)
    assert "spec.verify_early = vcfg.early_start_verify" in src and "spec.gate = lambda text" in src


# ------------------------------------------------------------------------------------------------ the opener the code picks
class Writes(Parallel):
    """Writes a reply that starts with its own "Sure," as models do."""

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        with self.lock:
            self.writes.append(user)
        yield "Sure, I have five Hyundais. "
        yield "Which one would you like to hear more about?"


def opener_engine(decisions, backend=Writes, opener_first=True):
    b = backend(decisions)
    e, _ = agent(None, backend=b)
    e.model_only = e.reply_writer.model_only = True
    e.opener_first = opener_first
    e.silent_fallback = False
    return e, b


def speak(e, s, text):
    out = list(e.handle_stream(s, text))
    return [x for x in out if isinstance(x, str)], out[-1]


def test_a_short_acknowledgement_is_said_first_and_the_models_own_one_is_dropped():
    e, _ = opener_engine(["search_cars"])
    lines, r = speak(e, e.new_session(), "do you have any Hyundais")
    assert lines[0] in OPENERS + NEUTRAL_OPENERS and lines[0].endswith(",")
    assert lines[1].startswith("I have five Hyundais") and "Sure" not in lines[1]      # no "Sure, ... Sure, ..."
    assert r.reply.startswith(lines[0]) and r.debug.intent == "search_cars"


def test_it_is_off_by_default_so_the_models_words_are_the_first_said():
    e, _ = opener_engine(["search_cars"], opener_first=False)
    lines, _ = speak(e, e.new_session(), "do you have any Hyundais")
    assert lines[0].startswith("Sure, I have five")


def test_it_is_said_for_a_car_question_and_not_for_a_goodbye_or_a_small_talk_turn():
    e, _ = opener_engine(["get_car", "end_call", "none"])
    s = e.new_session()
    lines, _ = speak(e, s, "tell me about the Land Cruiser")
    assert lines[0] in OPENERS + NEUTRAL_OPENERS
    for text in ("that's all, thanks, bye",):
        lines, _ = speak(e, e.new_session(), text)
        assert lines[0] not in OPENERS + NEUTRAL_OPENERS, lines


def test_a_reply_that_goes_ahead_during_the_hold_still_follows_the_opener():
    e, b = opener_engine(["search_cars"])
    e.hold_reply = True
    s, text = e.new_session(), "do you have any Hyundais"
    decision = e.predecide(s, text)
    spec = e.prepare_reply(s, text, decision)
    import concurrent.futures
    d, rep = concurrent.futures.Future(), concurrent.futures.Future()
    d.set_result(decision)
    rep.set_result(spec)
    out = list(e.handle_stream(s, text, predecided=d, reply_ahead=rep))
    lines = [x for x in out if isinstance(x, str)]
    assert lines[0] in OPENERS + NEUTRAL_OPENERS and "I have five Hyundais" in lines[1] and len(b.writes) == 1 and out[-1].debug.reply_ahead is True


def test_the_pipeline_switches_it_on_only_in_the_test_mode():
    from dealer_core.app import build_app
    from dealer_core.config import load_config
    from dealer_core.voice.pipeline import apply_voice_profile
    from .test_scale import Scripted

    for mo, on, want in (("true", "true", True), ("false", "true", False), ("true", "false", False)):
        cfg = load_config(environ={"VA__VOICE__MODEL_ONLY": mo, "VA__VOICE__OPENER_FIRST": on})
        b = Scripted()
        app = build_app(cfg, classifier=b, reply_mode="conversational", lead_backend=b, engine="agent")
        apply_voice_profile(app, cfg)
        assert app.engine.opener_first is want, (mo, on)


def test_the_recorder_is_told_whether_the_early_reading_matched_the_full_one():
    for early_text, full, kind in (("I'm looking for a pose.", "I'm looking for a Porsche.", "early_mismatch"), ("Do you have a Santa Fe?", "do you have a Santa Fe", "early_match")):
        sp, _ = spec_with(early_text, verify=True)
        events = []
        sp.on_event = lambda k, t: events.append(k)

        async def go():
            sp.start_early(b"\1\0" * 8000)
            sp.confirm()
            await sp.transcript()
            for _ in range(100):
                await asyncio.sleep(0.01)
                if sp._reply.done():
                    break
            sp.reply_for(full)

        asyncio.run(go())
        assert kind in events, (kind, events)
