"""28 Oct, quiet speech with whisper-small.en: the hint holds what the agent just said, and when the audio is weak the recogniser writes the hint down. "I want to come and
see it" came back as "Would you like to come and see it?", the agent's own question, and the agent answered that as if the caller had said it."""
import pytest

from dealer_core.voice.services import UNCLEAR, EchoGuard, echoes_agent

AGENT = "Okay, there's the Macan at 149,000 dirhams. Would you like to hear more about it or schedule a test drive?"
ASKED = "It's the 2016 Macan, black, with 91,000 kilometres. Would you like to come and see it?"


@pytest.mark.parametrize("heard,line", [("Would you like to come and see it?", ASKED), ("Would you like to hear more about the Macan?", AGENT),
                                         ("would you like to come and see it", ASKED)])
def test_the_agents_own_question_is_recognised(heard, line):
    assert echoes_agent(heard, line)


@pytest.mark.parametrize("heard,line", [("I want to come and see it.", ASKED), ("Yes, I want to come and see the Macan.", ASKED), ("I want to hear more about the Macan.", AGENT),
                                         ("Yes", ASKED), ("Would you", ASKED), ("What else can you tell me about the Macan?", AGENT), ("", ASKED)])
def test_what_a_caller_really_says_is_not(heard, line):
    assert not echoes_agent(heard, line)


class Engine:
    def __init__(self, with_line, without):
        self.with_line, self.without, self.prompts = with_line, without, []
        self.name = "fake"

    def transcribe(self, pcm, prompt=None):
        self.prompts.append(prompt)
        return (self.without if prompt == "plain" else self.with_line), 10.0


def guard(engine, line=ASKED):
    return EchoGuard(engine, lambda: line, lambda: "plain")


def test_an_echo_is_heard_again_without_the_agents_sentence_in_the_hint():
    e = Engine("Would you like to come and see it?", "I want to come and see it.")
    g = guard(e)
    text, ms = g.transcribe(b"\0\0", "cars, Macan. " + ASKED)
    assert text == "I want to come and see it." and e.prompts == ["cars, Macan. " + ASKED, "plain"] and ms == 20.0 and g.retries == 1


def test_an_echo_that_stays_an_echo_is_not_taken_as_the_callers_words():
    g = guard(Engine("Would you like to come and see it?", "Would you like to come and see it?"))
    assert g.transcribe(b"\0\0", "hint")[0] == UNCLEAR


def test_a_normal_answer_is_passed_through_with_one_run_only():
    e = Engine("I want to come and see it.", "unused")
    g = guard(e)
    assert g.transcribe(b"\0\0", "hint") == ("I want to come and see it.", 10.0) and e.prompts == ["hint"] and g.retries == 0


def test_the_guard_looks_like_the_engine_it_wraps():
    assert guard(Engine("x", "y")).name == "fake"


def test_the_pipeline_listens_through_the_guard_for_the_turn_and_for_the_early_reading():
    import inspect

    from dealer_core.voice import pipeline

    src = inspect.getsource(pipeline.build_pipeline)
    assert "OpenVinoWhisperSTTService(engine=hear" in src and "hear.transcribe(pcm, stt.prompt_for())" in src
    assert "parts.stt_engine.transcribe(" not in src


# ------------------------------------------------------------------------------------------------ C5: the plan's looser test was scored on the recorded turns and not adopted
def test_real_answers_that_share_words_with_the_agents_question_are_not_thrown_away():
    """ARCH_PLAN C5 proposed a plain overlap test (60% of the heard words in any sentence the agent said, 4+ words, no matching start). Scored on the 79 recorded segments
    (scripts/score_echo_guard.py) it flagged 4: the one real echo and these three real answers. The test in the code flags only the echo."""
    from dealer_core.voice.services import echoes_agent

    answers = [
        ("I want to know something else about the Macan.", "It uses about 9.8 litres per hundred kilometres. Would you like to know anything else about the Macan?"),
        ("I want to sure know about the Macan.", "It passed 149 of our 150 checks. Would you like to know anything else about the Macan?"),
        ("Two in the afternoon.", "The soonest is today: 2 in the afternoon or 5 in the evening. Which would you prefer?"),
    ]
    assert not any(echoes_agent(heard, said) for heard, said in answers)
    assert echoes_agent("Would you like to come and see it?", "It uses about 9.8 litres per hundred kilometres. Would you like to come and see it?")
