"""voice/heard.py: what the caller has heard of the agent's reply, from when its audio went out (owner, 29 Oct: the history kept the whole reply, spoken or not)."""
import numpy as np
from pipecat.frames.frames import AggregatedTextFrame, InterruptionFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection

from dealer_core.voice.heard import CUSHION_S, NETWORK_S, Playback, PlaybackTracker

LAG = CUSHION_S + NETWORK_S


def reply(p, at=0.0):
    p.sentence("There's the Corolla at 62,500 dirhams, the Camry at 74,500, and the RAV4 at 98,500.")
    for i in range(6):
        p.audio(1.0, at=at)                                                  # six seconds of audio, all handed over at once
    p.sentence("Which one would you like to hear more about?")
    p.audio(2.0, at=at)


def test_nothing_is_heard_before_the_page_starts_playing():
    p = Playback()
    reply(p)
    assert p.heard(at=LAG - 0.01) == "" and not p.playing(at=LAG - 0.01) and p.playing(at=LAG + 0.01)


def test_the_start_of_the_sentence_being_spoken_is_heard_never_more():
    p = Playback()
    reply(p)
    heard = p.heard(at=LAG + 2.0)                                            # a third of the first sentence
    assert heard.startswith("There's the Corolla") and heard.endswith(" ...") and "Camry" not in heard and "RAV4" not in heard


def test_the_audio_plays_back_to_back_so_a_sentence_sent_at_once_still_takes_its_length():
    p = Playback()
    reply(p)
    assert p.heard(at=LAG + 6.5) == "There's the Corolla at 62,500 dirhams, the Camry at 74,500, and the RAV4 at 98,500. Which one ..."
    assert p.playing(at=LAG + 7.9) and not p.playing(at=LAG + 8.1)


def test_a_number_counts_as_long_as_it_takes_to_say():
    p = Playback()
    p.sentence("It is 189,000 dirhams now.")
    p.audio(4.0, at=0.0)
    assert "189,000" not in p.heard(at=LAG + 0.6)                             # "It is one hundred and eighty nine thousand": not done after 0.6 of 4 s


def test_the_tracker_tells_the_brain_what_was_heard_when_the_caller_interrupts():
    cuts = []
    clock = [0.0]
    t = PlaybackTracker(Playback(clock=lambda: clock[0]), on_cut=cuts.append)
    pcm = np.zeros(24000, dtype=np.int16).tobytes()                          # 1 s at 24 kHz
    t.observe(AggregatedTextFrame("There's the Macan at 149,000 dirhams.", text_type="sentence"))
    for _ in range(3):
        t.observe(TTSAudioRawFrame(audio=pcm, sample_rate=24000, num_channels=1))
    clock[0] = LAG + 3.5
    t.observe(InterruptionFrame())
    assert cuts == ["There's the Macan at 149,000 dirhams."]
    t.observe(InterruptionFrame())                                           # nothing is playing any more: nothing to report
    assert len(cuts) == 1
