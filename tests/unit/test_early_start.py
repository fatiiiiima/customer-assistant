"""Start recognising and deciding at 0.2 s of quiet inside a turn, not only when Smart Turn holds a pause. In tap mode the voice detector waits 0.6 s to call the turn over,
so this answers up to 0.4 s sooner. Used as it is if the turn ends in that quiet, dropped if the caller speaks again."""
import asyncio

import numpy as np
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.frames.frames import InputAudioRawFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

from dealer_core.voice.endpoint import SmartEndpointer
from tests.unit.test_endpoint import FakeAnalyzer
from tests.unit.test_speculate import Recorder

FRAME = 320                                        # 20 ms at 16 kHz


def loud(amplitude=8000):
    return InputAudioRawFrame(audio=(np.sin(np.arange(FRAME) / 3) * amplitude).astype("<i2").tobytes(), sample_rate=16000, num_channels=1)


def quiet(amplitude=20):
    return InputAudioRawFrame(audio=(np.random.default_rng(1).normal(0, amplitude, FRAME)).astype("<i2").tobytes(), sample_rate=16000, num_channels=1)


def started():
    return VADUserStartedSpeakingFrame(start_secs=0.1)


def stopped():
    return VADUserStoppedSpeakingFrame(stop_secs=0.6)


def run(frames, early=0.2, verdicts=(EndOfTurnState.COMPLETE,)):
    ep = SmartEndpointer(analyzer=FakeAnalyzer(verdicts), early_start_secs=early)
    ep.speculation = Recorder()
    out = []

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        out.append(type(frame).__name__)
    ep.push_frame = push

    async def go():
        for f in frames:
            await ep.process_frame(f, FrameDirection.DOWNSTREAM)
    asyncio.run(go())
    return ep, ep.speculation.calls, [o for o in out if o != "InputAudioRawFrame"]


def words(n=10):
    return [loud() for _ in range(n)]


def test_work_starts_after_the_quiet_and_is_confirmed_when_the_turn_ends_in_it():
    ep, calls, passed = run([started(), *words(), *[quiet() for _ in range(30)], stopped()])         # 0.6 s of quiet, like tap mode
    assert [c[0] for c in calls if c[0] != "clear"] == ["start", "confirm"] and ep.early_starts == 1
    assert passed == ["VADUserStartedSpeakingFrame", "VADUserStoppedSpeakingFrame"]


def test_the_work_starts_at_0_2_s_not_before():
    _, calls, _ = run([started(), *words(), *[quiet() for _ in range(9)]])                        # 0.18 s
    assert [c for c in calls if c[0] == "start"] == []
    _, calls, _ = run([started(), *words(), *[quiet() for _ in range(10)]])                       # 0.20 s
    assert [c[0] for c in calls if c[0] == "start"] == ["start"]


def test_the_caller_speaking_again_drops_the_work():
    ep, calls, _ = run([started(), *words(), *[quiet() for _ in range(12)], *words(5), *[quiet() for _ in range(30)], stopped()])
    kinds = [c[0] for c in calls if c[0] != "clear"]
    assert kinds == ["start", "resumed", "start", "confirm"] and ep.early_starts == 2


def test_a_soft_word_is_not_mistaken_for_quiet():
    """About 20 dB below the turn's peak is quiet speech, not silence: nothing is started."""
    soft = [loud(900) for _ in range(30)]                                                   # peak 8000 -> 900 is 19 dB down: above the 0.06 share
    _, calls, _ = run([started(), *words(), *soft])
    assert [c for c in calls if c[0] == "start"] == []


def test_room_noise_after_a_quiet_voice_still_counts_as_quiet():
    ep, calls, _ = run([started(), *[loud(2000) for _ in range(10)], *[quiet(60) for _ in range(12)]])
    assert [c[0] for c in calls if c[0] == "start"] == ["start"]


def test_at_most_three_starts_in_a_turn():
    frames = [started()]
    for _ in range(6):
        frames += [*words(4), *[quiet() for _ in range(12)]]
    ep, calls, _ = run(frames)
    assert ep.early_starts == 3 and sum(1 for c in calls if c[0] == "start") == 3


def test_it_is_off_when_the_setting_is_zero():
    _, calls, _ = run([started(), *words(), *[quiet() for _ in range(40)], stopped()], early=0.0)
    assert [c[0] for c in calls if c[0] != "clear"] == []


def test_a_held_pause_still_starts_its_own_work_and_is_not_confirmed_at_the_stop():
    """Smart Turn says the sentence is unfinished: the existing hold path takes over (start at the hold, confirm only when its own fallback silence passes)."""
    ep, calls, passed = run([started(), *words(), *[quiet() for _ in range(12)], stopped()], verdicts=(EndOfTurnState.INCOMPLETE,))
    assert [c[0] for c in calls if c[0] != "clear"] == ["start", "start"]                    # the early one, then the hold's own
    assert passed == ["VADUserStartedSpeakingFrame"] and ep.held == 1


def test_a_new_turn_starts_from_scratch():
    ep, calls, _ = run([started(), *words(), *[quiet() for _ in range(12)], stopped(), started(), *[loud(1000) for _ in range(10)], *[quiet() for _ in range(12)], stopped()],
                       verdicts=(EndOfTurnState.COMPLETE, EndOfTurnState.COMPLETE))
    assert [c[0] for c in calls if c[0] != "clear"] == ["start", "confirm", "start", "confirm"]


def test_without_a_speculation_nothing_happens_and_nothing_breaks():
    ep = SmartEndpointer(analyzer=FakeAnalyzer([EndOfTurnState.COMPLETE]), early_start_secs=0.2)

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        pass
    ep.push_frame = push

    async def go():
        for f in [started(), *words(), *[quiet() for _ in range(30)], stopped()]:
            await ep.process_frame(f, FrameDirection.DOWNSTREAM)
    asyncio.run(go())
    assert ep.early_starts == 0


def test_the_setting_and_its_wiring():
    import inspect

    from dealer_core.config import load_config
    from dealer_core.voice import pipeline

    assert load_config(environ={}).voice.early_start_secs == 0          # off by default: its slices were heard worse than the normal path (docs/ARCH_REVIEW.md section 4, 10)
    assert load_config(environ={"VA__VOICE__EARLY_START_SECS": "0.2"}).voice.early_start_secs == 0.2
    assert "early_start_secs=vcfg.early_start_secs if vcfg.speculate else 0.0" in inspect.getsource(pipeline.build_pipeline)
