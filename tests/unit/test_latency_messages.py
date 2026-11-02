"""The messages behind the page's latency table (server/latency.py, ARCH_REVIEW section 88), and the end-of-turn check time the endpointer keeps."""
import asyncio

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.frames.frames import (
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from dealer_core.dialogue.machine import TurnDebug, TurnResult
from dealer_core.server.latency import detail_message, timing_message
from dealer_core.voice.endpoint import SmartEndpointer
from dealer_core.voice.status import TurnClock

from .test_endpoint import FakeAnalyzer

PARTS = {"waiting for you to finish": 0.2, "speech recognition": 0.121, "reply": 0.754, "voice": 0.138}


def test_the_timing_message_carries_every_step_and_a_total_that_includes_the_check():
    m = timing_message(1.213, PARTS, 0.047, 3)
    assert m == {"type": "timing", "seconds": 1.26, "turn": 3, "wait": 0.2, "check": 0.047, "hear": 0.121, "reply": 0.754, "voice": 0.138}


def test_a_step_the_turn_did_not_have_is_none_and_a_negative_check_is_zero():
    m = timing_message(0.4, {"waiting for you to finish": 0.2}, -0.01, 1)
    assert m["hear"] is None and m["reply"] is None and m["voice"] is None and m["check"] == 0.0 and m["seconds"] == 0.4


def test_the_turn_detail_has_the_decision_the_parallel_hit_and_the_size_of_the_reply():
    d = TurnDebug(intent="get_car", nlu_ms=512.4, reply_ahead=True)
    assert detail_message(TurnResult(reply="It is the white 2021 Camry.", debug=d), 4) == {"type": "turn_detail", "turn": 4, "decision": 0.512, "ahead": True, "tool": "get_car", "words": 6}
    off = detail_message(TurnResult(reply="Hello", debug=TurnDebug()), 1)
    assert off["ahead"] is None and off["decision"] is None and off["tool"] is None


def test_the_clocks_parts_flow_through_to_the_message():
    got = []
    t = [0.0]
    clock = TurnClock(0.2, lambda total, parts: got.append((total, parts)), clock=lambda: t[0])
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        TranscriptionFrame,
        TTSSpeakFrame,
    )

    def at(now, frame, stage):
        t[0] = now
        clock.see(frame, stage)

    at(10.0, VADUserStoppedSpeakingFrame(stop_secs=0.2), "vad")
    at(10.12, TranscriptionFrame("Camry", "u", "t"), "stt")
    at(10.87, TTSSpeakFrame("It is the Camry."), "brain")
    at(11.0, BotStartedSpeakingFrame(), "vad")
    total, parts = got[0]
    msg = timing_message(total, parts, 0.05, 2)
    assert (msg["wait"], msg["hear"], msg["reply"], msg["voice"]) == (0.2, 0.12, 0.75, 0.13) and msg["seconds"] == 1.25


def run(frames, verdicts):
    ep = SmartEndpointer(analyzer=FakeAnalyzer(verdicts, fallback_after=None))
    out = []

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        out.append(frame)
    ep.push_frame = push

    async def go():
        for f in frames:
            await ep.process_frame(f, FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.01)
    asyncio.run(go())
    return ep


def audio():
    return InputAudioRawFrame(audio=b"\0" * 640, sample_rate=16000, num_channels=1)


def test_the_endpointer_keeps_how_long_it_held_the_turn_open():
    ep = run([VADUserStartedSpeakingFrame(start_secs=0.1), audio(), VADUserStoppedSpeakingFrame(stop_secs=0.2)], [EndOfTurnState.COMPLETE])
    assert 0.0 <= ep.last_check_secs < 0.5                      # a verdict at once: only the check itself
    held = SmartEndpointer(analyzer=FakeAnalyzer([EndOfTurnState.INCOMPLETE], fallback_after=3))
    out = []

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        out.append(type(frame).__name__)
    held.push_frame = push

    async def go():
        for f in (VADUserStartedSpeakingFrame(start_secs=0.1), audio(), VADUserStoppedSpeakingFrame(stop_secs=0.2)):
            await held.process_frame(f, FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.15)                               # the fallback silence passes: three quiet frames
        for _ in range(3):
            await held.process_frame(audio(), FrameDirection.DOWNSTREAM)
    asyncio.run(go())
    assert out.count("VADUserStoppedSpeakingFrame") == 1 and held.last_check_secs >= 0.14
