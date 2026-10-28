"""28 Oct: very quiet or whispered speech reaches the recogniser as sound it cannot make out, and it writes what it learned from video captions ("Please subscribe and hit
the bell icon", "Please feel free to ask in the comments below", a lone "Okay.") or a near miss; the agent then acted on that. Now a very quiet turn, or a quiet one whose words
look invented, is answered with a request to speak up. The numbers behind the settings are in docs/ARCH_REVIEW.md section 13."""
import asyncio

import pytest
from pipecat.frames.frames import TTSSpeakFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

from dealer_core.config import load_config
from dealer_core.voice.brain import DealerBrain
from dealer_core.voice.services import SPEAK_UP, too_quiet_to_trust

DOWN = FrameDirection.DOWNSTREAM
VERY, SUSPECT = 0.02, 0.05


# ------------------------------------------------------------------------------------------------ the decision
@pytest.mark.parametrize("text,level", [("I want to hear more about the Macan.", 0.015), ("What else can you tell me about the Macan?", 0.011), ("Yes", 0.01),
                                         ("Looking for the fish.", 0.019)])
def test_a_very_quiet_turn_is_not_trusted_whatever_the_words(text, level):
    assert too_quiet_to_trust(text, level, VERY, SUSPECT)


@pytest.mark.parametrize("text", ["Please subscribe and hit the bell icon so you can be notified.", "Please feel free to ask in the comments below.", "Okay.", "Thank you.",
                                  "Thanks for watching!", "you"])
def test_a_quiet_turn_with_invented_looking_words_is_not_trusted(text):
    assert too_quiet_to_trust(text, 0.043, VERY, SUSPECT)


@pytest.mark.parametrize("text,level", [("I want to hear more about the Macan.", 0.03), ("I want to share more about the Macan.", 0.045), ("Yes, talk about the Macan.", 0.021),
                                         ("No, I want to know something else about the Macan.", 0.04), ("Looking for the fish.", 0.021)])
def test_a_quiet_turn_with_ordinary_words_is_acted_on(text, level):
    assert not too_quiet_to_trust(text, level, VERY, SUSPECT)          # (a wrong word like 'fish' cannot be told from the words alone: this is the part not caught)


@pytest.mark.parametrize("text", ["Please subscribe and hit the bell icon.", "Okay.", "Thank you.", "I want to hear more about the Macan."])
def test_a_normal_voice_is_never_stopped(text):
    assert not too_quiet_to_trust(text, 0.06, VERY, SUSPECT) and not too_quiet_to_trust(text, 0.2, VERY, SUSPECT)


def test_no_words_no_level_or_the_setting_at_zero_never_stops_a_turn():
    assert not too_quiet_to_trust("", 0.001, VERY, SUSPECT) and not too_quiet_to_trust("hello", None, VERY, SUSPECT)
    assert not too_quiet_to_trust("hello", 0.001, 0, 0) and not too_quiet_to_trust("Okay.", 0.03, VERY, 0)
    assert too_quiet_to_trust("hello", 0.001, VERY, 0) 
    assert too_quiet_to_trust("Okay.", 0.03, 0, SUSPECT)                # each part works alone


def test_the_settings_and_the_way_to_switch_them_off():
    v = load_config(environ={}).voice
    assert v.speak_up_below == 0.02 and v.speak_up_suspect_below == 0.05
    off = load_config(environ={"VA__VOICE__SPEAK_UP_BELOW": "0", "VA__VOICE__SPEAK_UP_SUSPECT_BELOW": "0"}).voice
    assert off.speak_up_below == 0 and off.speak_up_suspect_below == 0
    assert SPEAK_UP == "Sorry, I can barely hear you. Could you speak up or move closer to the microphone?"


# ------------------------------------------------------------------------------------------------ the brain
class Turn:
    def __init__(self):
        self.debug = type("D", (), {"intent": "none", "via": "rules", "draft": False, "risk": "low", "note": ""})()
        self.ended = False


class Session:
    def __init__(self):
        self.history = []

    def remember(self, who, text):
        self.history.append((who, text))


class Engine:
    def __init__(self):
        self.seen = []

    def new_session(self):
        return Session()

    def handle_stream(self, s, text, **extra):
        self.seen.append(text)
        yield f"Reply to: {text}"
        s.remember("customer", text)
        yield Turn()


def make(level):
    levels = {"now": level}
    said, heard, pushed = [], [], []
    brain = DealerBrain(Engine(), speak_opening=False, on_heard=heard.append, on_say=said.append, level_of=lambda: levels["now"], speak_up=(VERY, SUSPECT))

    async def push(frame, direction=DOWN):
        pushed.append(frame)
    brain.push_frame = push
    return brain, levels, said, heard, pushed


def hear(brain, text):
    return brain.process_frame(TranscriptionFrame(text, "user", "2026-10-28T11:00:00Z"), DOWN)


def test_a_very_quiet_turn_gets_the_request_and_the_engine_is_not_asked():
    async def go():
        brain, levels, said, heard, pushed = make(0.012)
        await hear(brain, "I want to hear more about the Macan.")
        return brain, said, heard, pushed
    brain, said, heard, pushed = asyncio.run(go())
    assert brain.engine.seen == [] and said == [SPEAK_UP] and heard == []                    # nothing is acted on, and the garbage is not shown as what the caller said
    assert [f.text for f in pushed if isinstance(f, TTSSpeakFrame)] == [SPEAK_UP]
    assert brain.session.history == [] and brain.spoken_up == 1                              # ...and not remembered, so the conversation goes on from where it was


def test_a_normal_turn_goes_to_the_engine_as_before():
    async def go():
        brain, levels, said, heard, pushed = make(0.09)
        await hear(brain, "I want to hear more about the Macan.")
        return brain, said, heard
    brain, said, heard = asyncio.run(go())
    assert brain.engine.seen == ["I want to hear more about the Macan."] and said == ["Reply to: I want to hear more about the Macan."] and brain.spoken_up == 0


def test_after_two_requests_in_a_row_the_words_are_used_so_nobody_is_stuck():
    async def go():
        brain, levels, said, heard, pushed = make(0.012)
        for text in ("one one one", "two two two", "three three three", "four four four"):
            await hear(brain, text)
        return brain, said
    brain, said = asyncio.run(go())
    assert said[:2] == [SPEAK_UP, SPEAK_UP] and brain.engine.seen[:1] == ["three three three"]      # the third is taken as it is
    assert brain.spoken_up == 3                                                                      # (the fourth is asked again: a new run of two)


def test_a_clear_turn_in_between_starts_the_count_again():
    async def go():
        brain, levels, said, heard, pushed = make(0.012)
        await hear(brain, "whisper one")
        levels["now"] = 0.1
        await hear(brain, "Tell me about the Macan.")
        levels["now"] = 0.012
        await hear(brain, "whisper two")
        await hear(brain, "whisper three")
        return said
    said = asyncio.run(go())
    assert said.count(SPEAK_UP) == 3


def test_with_the_setting_at_zero_the_brain_never_asks():
    async def go():
        brain, levels, said, heard, pushed = make(0.001)
        brain._speak_up = (0.0, 0.0)
        await hear(brain, "hello there")
        return brain
    assert asyncio.run(go()).engine.seen == ["hello there"]


def test_the_pipeline_passes_the_settings_to_the_brain():
    import inspect

    from dealer_core.voice import pipeline

    assert "speak_up=(vcfg.speak_up_below, vcfg.speak_up_suspect_below)" in inspect.getsource(pipeline.build_pipeline)
