"""What the call recorder (voice/audiotap.py, only when VA_DUMP_AUDIO_DIR is set) now writes beside the audio so a live call can be read afterwards (ARCH_REVIEW section 83):
why a speech over the agent did or did not stop it, and one line per turn with how it was decided and how long each part took."""
import json

import numpy as np
from pipecat.frames.frames import (
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from dealer_core.dialogue.machine import TurnDebug, TurnResult
from dealer_core.voice.bargein import BargeIn, LateBargeIn
from dealer_core.voice.pipeline import turn_trace


class Audible:
    def __init__(self, on=True):
        self.on = on

    def playing(self, at=None):
        return self.on

    def heard(self):
        return "Which one"

    def new_reply(self):
        pass


def audio(level, secs=0.02):
    n = int(16000 * secs)
    return InputAudioRawFrame(audio=(np.sin(np.arange(n) / 3) * level * 1.414 * 32767).astype(np.int16).tobytes(), sample_rate=16000, num_channels=1)


def feed(b, frames):
    return [b.observe(f, FrameDirection.DOWNSTREAM) for f in frames]


def test_a_cut_is_noted_with_how_much_speech_it_took_and_how_loud():
    events = []
    b = BargeIn(Audible(), min_speech_secs=0.3, on_event=lambda k, t: events.append((k, t)))
    feed(b, [VADUserStartedSpeakingFrame(start_secs=0.1)] + [audio(0.05)] * 20)
    assert [k for k, _ in events] == ["cut"] and "0.30 s" in events[0][1] and "loudest 0.050" in events[0][1]


def test_a_speech_over_the_agent_that_was_too_short_or_too_quiet_says_so():
    events = []
    b = BargeIn(Audible(), min_speech_secs=0.3, level=0.01, on_event=lambda k, t: events.append((k, t)))
    feed(b, [VADUserStartedSpeakingFrame(start_secs=0.1)] + [audio(0.05)] * 8 + [VADUserStoppedSpeakingFrame(stop_secs=0.3)])
    feed(b, [VADUserStartedSpeakingFrame(start_secs=0.1)] + [audio(0.004)] * 30 + [VADUserStoppedSpeakingFrame(stop_secs=0.3)])
    assert [k for k, _ in events] == ["over_agent", "over_agent"]
    assert "0.16 s" in events[0][1] and "loudest 0.050" in events[0][1]
    assert "0.00 s" in events[1][1] and "loudest 0.004" in events[1][1]            # the caller's speech never reached the level while the agent was audible


def test_the_agent_finishing_first_is_noted():
    events = []
    p = Audible()
    b = BargeIn(p, on_event=lambda k, t: events.append((k, t)))
    feed(b, [VADUserStartedSpeakingFrame(start_secs=0.1)] + [audio(0.05)] * 5)
    p.on = False
    feed(b, [audio(0.05)])
    assert [k for k, _ in events] == ["over_agent"] and "finished by itself" in events[0][1]


def test_no_recorder_means_no_change_in_the_decision():
    b = BargeIn(Audible(), min_speech_secs=0.3)
    assert feed(b, [VADUserStartedSpeakingFrame(start_secs=0.1)] + [audio(0.05)] * 20).count(True) == 1


class Brain:
    def __init__(self, yes):
        self.yes, self.cut = yes, []

    def talks_over(self, text):
        return self.yes

    def reply_cut(self, heard):
        self.cut.append(heard)


def words(text):
    return TranscriptionFrame(text=text, user_id="u", timestamp="t")


def test_words_that_arrive_over_the_agent_are_noted_whether_or_not_they_stopped_it():
    events = []
    late = LateBargeIn(Brain(True), Audible(), on_event=lambda k, t: events.append((k, t)))
    assert late.observe(words("the Macan")) is True
    kept = LateBargeIn(Brain(False), Audible(), on_event=lambda k, t: events.append((k, t)))
    assert kept.observe(words("okay")) is False
    assert events == [("late_cut", "stopped the agent for the words 'the Macan'"), ("late_kept", "words arrived over the agent but were not taken as talking over it: 'okay'")]


def test_a_turn_line_carries_how_it_was_decided_and_the_times():
    d = TurnDebug(intent="ask_price", via="llm:bedrock_nemotron", state_after="answering", nlu_ms=412.4, lead_ms=1180.6, total_ms=1630.2, note="x" * 900)
    line = json.loads(turn_trace(TurnResult(reply="It is 189,000 dirhams. Want to book a test drive?", debug=d)))
    assert line["intent"] == "ask_price" and line["via"] == "llm:bedrock_nemotron" and line["state"] == "answering"
    assert (line["decision_ms"], line["first_piece_ms"], line["turn_ms"]) == (412, 1181, 1630) and line["words"] == 10 and line["ended"] is False
    assert len(line["note"]) == 700                                                  # the engine's note is kept, cut to a line
