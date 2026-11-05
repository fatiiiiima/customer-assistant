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


# ------------------------------------------------------------------------------------------------ late barge-in (LateBargeIn)
def late_setup(level=0.2, caller_levels=(0.2,)):
    """A brain that has answered "Any Toyotas?" and a page 1.5 s into playing that reply."""
    import asyncio

    from pipecat.frames.frames import TranscriptionFrame

    from dealer_core.voice.bargein import LateBargeIn
    from dealer_core.voice.heard import Playback

    from .test_barge_in_history import LIST, QUESTION, Engine, brain_for

    b = brain_for(Engine())
    asyncio.run(b._answer("Any Toyotas?"))
    b._level_of, b._background_below, b._caller_levels = (lambda: level), 0.25, list(caller_levels)
    now = [100.0]
    pb = Playback(clock=lambda: now[0])
    for s in (LIST, QUESTION):
        pb.sentence(s)
        pb.audio(4.0, at=100.0)
    now[0] = 101.5
    b.playback, late = pb, LateBargeIn(b, pb)
    return b, pb, late, lambda text: late.observe(TranscriptionFrame(text, "caller", "t"))


def test_a_short_word_said_over_the_agent_stops_it_when_its_words_arrive():
    """30 Oct, the owner: "it hears me but it continues speaking the full response, my barge in response gets printed, and then it replies to that barge in".
    In recorded hands-free calls 47 of 132 stretches of speech over the agent did not stop it; "Tomorrow.", "no", "Porsche." and "Test drive." among them."""
    for said in ("Tomorrow.", "no", "Porsche.", "Do you have any BMWs?"):
        b, pb, late, hear = late_setup()
        assert hear(said), said
        assert b.session.history[-1][1].endswith("(interrupted)") and "RAV4" not in b.session.history[-1][1]     # only what the caller heard is kept
        assert not pb.playing() and late.interruptions == 1


def test_a_yes_a_thank_you_the_room_or_the_agents_own_words_do_not_stop_it():
    from .test_barge_in_history import QUESTION

    for said in ("Yeah.", "okay", "Thank you.", "Mm-hmm.", "um", "Thanks for watching!", QUESTION):
        _, pb, late, hear = late_setup()
        assert not hear(said) and pb.playing() and late.interruptions == 0, said
    _, pb, late, hear = late_setup(level=0.03)                                     # under a quarter of the caller's usual level: someone else in the room
    assert not hear("I'm going to drive a little bit.")


def test_nothing_to_stop_once_the_reply_has_finished_playing():
    b, pb, _, hear = late_setup()
    pb.clock = lambda: 120.0
    assert not hear("Tomorrow.") and b.session.history[-1][1].endswith("?")
