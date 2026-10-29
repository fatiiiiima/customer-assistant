"""Barge-in (owner, 29 Oct: "while the model is speaking, if I interrupt it, it is able to understand and changes the response"). The agent stops only for real
speech while it is audible on the page: not for a cough or a click, not for quiet echo, not once it has finished. These exercise BargeIn.observe, the real decision."""
import numpy as np
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InputAudioRawFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

from dealer_core.voice.bargein import BargeIn


class Audible:
    def __init__(self, on=True):
        self.on = on

    def playing(self, at=None):
        return self.on


def audio(level, secs=0.02):
    n = int(16000 * secs)
    return InputAudioRawFrame(audio=(np.sin(np.arange(n) / 3) * level * 1.414 * 32767).astype(np.int16).tobytes(), sample_rate=16000, num_channels=1)


def feed(b, frames):
    return [b.observe(f, FrameDirection.UPSTREAM if isinstance(f, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)) else FrameDirection.DOWNSTREAM) for f in frames]


def test_speech_over_the_audible_agent_interrupts_after_about_three_tenths_of_a_second():
    b = BargeIn(Audible(), min_speech_secs=0.3)
    out = feed(b, [VADUserStartedSpeakingFrame(start_secs=0.1)] + [audio(0.05)] * 20)
    assert out.index(True) == 15 and b.interruptions == 1                     # 15 frames of 20 ms = 0.3 s
    assert out.count(True) == 1


def test_a_cough_or_a_click_does_not():
    b = BargeIn(Audible(), min_speech_secs=0.3)
    out = feed(b, [VADUserStartedSpeakingFrame(start_secs=0.1)] + [audio(0.05)] * 8 + [VADUserStoppedSpeakingFrame(stop_secs=0.3)] + [audio(0.05)] * 20)
    assert not any(out) and b.ignored == 1


def test_quiet_echo_of_the_agent_does_not():
    b = BargeIn(Audible(), min_speech_secs=0.3, level=0.01)
    assert not any(feed(b, [VADUserStartedSpeakingFrame(start_secs=0.1)] + [audio(0.004)] * 100))


def test_nothing_is_interrupted_when_the_agent_is_not_audible_or_has_just_finished():
    assert not any(feed(BargeIn(Audible(False)), [VADUserStartedSpeakingFrame(start_secs=0.1)] + [audio(0.05)] * 30))
    p = Audible(True)
    b = BargeIn(p)
    feed(b, [VADUserStartedSpeakingFrame(start_secs=0.1)] + [audio(0.05)] * 5)
    p.on = False                                                             # the reply ended while the caller started
    assert not any(feed(b, [audio(0.05)] * 30))


def test_the_page_s_playback_counts_not_the_server_s_sending():
    """The server sends audio at twice its speed: after its 'bot stopped speaking' the page is still playing, and the caller can still interrupt."""
    b = BargeIn(Audible(True))
    assert any(feed(b, [BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(), VADUserStartedSpeakingFrame(start_secs=0.1)] + [audio(0.05)] * 20))


def test_without_a_playback_model_the_server_s_view_is_used():
    b = BargeIn(None)
    assert not any(feed(b, [VADUserStartedSpeakingFrame(start_secs=0.1)] + [audio(0.05)] * 20))
    assert any(feed(b, [BotStartedSpeakingFrame(), VADUserStartedSpeakingFrame(start_secs=0.1)] + [audio(0.05)] * 20))
