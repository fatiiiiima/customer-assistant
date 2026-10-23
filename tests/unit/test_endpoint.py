"""SmartEndpointer: a stop is let through only when the turn model says complete (or the fallback silence passes); a held stop swallows the next start."""
import asyncio

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.frames.frames import InputAudioRawFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

from dealer_core.voice.endpoint import SmartEndpointer


class FakeAnalyzer:
    def __init__(self, verdicts, fallback_after=None):
        self.verdicts, self.fallback_after, self.silent = list(verdicts), fallback_after, 0

    def set_sample_rate(self, rate):
        pass

    async def analyze_end_of_turn(self):
        return self.verdicts.pop(0), None

    def append_audio(self, buf, is_speech):
        self.silent = 0 if is_speech else self.silent + 1
        return EndOfTurnState.COMPLETE if self.fallback_after and self.silent >= self.fallback_after else EndOfTurnState.INCOMPLETE


def run(analyzer, frames):
    ep = SmartEndpointer(analyzer=analyzer)
    out = []

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        out.append(frame)
    ep.push_frame = push

    async def go():
        for f in frames:
            await ep.process_frame(f, FrameDirection.DOWNSTREAM)
    asyncio.run(go())
    return ep, [type(f).__name__ for f in out if not isinstance(f, InputAudioRawFrame)]


def audio():
    return InputAudioRawFrame(audio=b"\0" * 640, sample_rate=16000, num_channels=1)


def test_complete_turn_passes_through():
    _, seen = run(FakeAnalyzer([EndOfTurnState.COMPLETE]), [VADUserStartedSpeakingFrame(start_secs=0.2), audio(), VADUserStoppedSpeakingFrame(stop_secs=0.2)])
    assert seen == ["VADUserStartedSpeakingFrame", "VADUserStoppedSpeakingFrame"]


def test_incomplete_turn_is_held_and_the_continuation_is_one_segment():
    ep, seen = run(FakeAnalyzer([EndOfTurnState.INCOMPLETE, EndOfTurnState.COMPLETE]),
                   [VADUserStartedSpeakingFrame(start_secs=0.2), audio(), VADUserStoppedSpeakingFrame(stop_secs=0.2), audio(),
                    VADUserStartedSpeakingFrame(start_secs=0.2), audio(), VADUserStoppedSpeakingFrame(stop_secs=0.2)])
    assert seen == ["VADUserStartedSpeakingFrame", "VADUserStoppedSpeakingFrame"] and ep.held == 1


def test_fallback_silence_ends_a_held_turn():
    _, seen = run(FakeAnalyzer([EndOfTurnState.INCOMPLETE], fallback_after=3),
                  [VADUserStartedSpeakingFrame(start_secs=0.2), audio(), VADUserStoppedSpeakingFrame(stop_secs=0.2), audio(), audio(), audio(), audio()])
    assert seen == ["VADUserStartedSpeakingFrame", "VADUserStoppedSpeakingFrame"]


def audio_secs(secs):
    return InputAudioRawFrame(audio=b"\0" * int(16000 * secs * 2), sample_rate=16000, num_channels=1)


def run_ctx(analyzer, frames, asked):
    ep = SmartEndpointer(analyzer=analyzer, short_answer_secs=1.4)
    ep.asked_question = (lambda: asked) if asked is not None else None
    out = []

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        out.append(frame)
    ep.push_frame = push

    async def go():
        for f in frames:
            await ep.process_frame(f, FrameDirection.DOWNSTREAM)
    asyncio.run(go())
    return ep, [type(f).__name__ for f in out if not isinstance(f, InputAudioRawFrame)]


def test_a_short_answer_right_after_a_question_is_finished_even_if_smart_turn_hesitates():
    ep, seen = run_ctx(FakeAnalyzer([EndOfTurnState.INCOMPLETE]),
                       [VADUserStartedSpeakingFrame(start_secs=0.1), audio_secs(0.6), VADUserStoppedSpeakingFrame(stop_secs=0.2)], asked=True)
    assert seen == ["VADUserStartedSpeakingFrame", "VADUserStoppedSpeakingFrame"] and ep.short_answers == 1


def test_a_long_utterance_or_no_question_is_still_held_when_smart_turn_hesitates():
    _, seen = run_ctx(FakeAnalyzer([EndOfTurnState.INCOMPLETE]),
                      [VADUserStartedSpeakingFrame(start_secs=0.1), audio_secs(3.0), VADUserStoppedSpeakingFrame(stop_secs=0.2)], asked=True)
    assert seen == ["VADUserStartedSpeakingFrame"]                      # 3 s of speech is not a short answer
    _, seen = run_ctx(FakeAnalyzer([EndOfTurnState.INCOMPLETE]),
                      [VADUserStartedSpeakingFrame(start_secs=0.1), audio_secs(0.6), VADUserStoppedSpeakingFrame(stop_secs=0.2)], asked=False)
    assert seen == ["VADUserStartedSpeakingFrame"]                      # the agent had not asked anything


# ---- the words rule: a complete answer to the agent's question ends the turn when Smart Turn hesitates -------------------------------------------
def run_peek(text, is_answer):
    ep = SmartEndpointer(analyzer=FakeAnalyzer([EndOfTurnState.INCOMPLETE]), short_answer_secs=0.0)

    async def peek(pcm):
        return text
    ep.peek, ep.is_answer = peek, is_answer
    out = []

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        out.append(frame)
    ep.push_frame = push

    async def go():
        for f in [VADUserStartedSpeakingFrame(start_secs=0.1), audio_secs(0.6), VADUserStoppedSpeakingFrame(stop_secs=0.2)]:
            await ep.process_frame(f, FrameDirection.DOWNSTREAM)
    asyncio.run(go())
    return ep, [type(f).__name__ for f in out if not isinstance(f, InputAudioRawFrame)]


def test_when_smart_turn_hesitates_a_complete_answer_ends_the_turn():
    ep, seen = run_peek("Toyota.", lambda t: True)
    assert seen == ["VADUserStartedSpeakingFrame", "VADUserStoppedSpeakingFrame"] and ep.short_answers == 1 and ep.peeked == 1


def test_when_smart_turn_hesitates_an_unfinished_sentence_is_still_held():
    ep, seen = run_peek("Can I book", lambda t: False)
    assert seen == ["VADUserStartedSpeakingFrame"] and ep.peeked == 1
