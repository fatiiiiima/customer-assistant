"""The caller speaks first, and silence on the line (owner, 30 Oct; docs/ARCH_REVIEW.md section 49):
* the greeting is said only after `greet_after` s of quiet at the start;
* after the agent has finished, `still_there_after` s of quiet gets "Hello? Are you still there?";
* `hang_up_after` s more gets the goodbye, and once it has been heard the call is ended (and kept, so the visitor can carry on).
Timings are shrunk here (the real ones: 3 s, 8 s, 7 s)."""
import asyncio
import sys
from pathlib import Path

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, TTSSpeakFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

sys.path.insert(0, str(Path(__file__).parent))

from test_speak_up import Engine, hear  # noqa: E402

from dealer_core.config import load_config  # noqa: E402
from dealer_core.voice.brain import SILENT_GOODBYE, STILL_THERE_ASK, WELCOME_BACK, DealerBrain  # noqa: E402

DOWN, UP = FrameDirection.DOWNSTREAM, FrameDirection.UPSTREAM
T = 0.05                                                     # one "second" of the real settings, shrunk


class Greeter(Engine):
    def opening(self, s):
        line = "Hello, I'm Wazir from AGMC. What kind of car are you after?"
        s.remember("agent", line)
        return line


def make(**kw):
    said, ended, greeted = [], [], []
    settings = dict(greet_after=3 * T, still_there_after=8 * T, hang_up_after=7 * T)
    settings.update(kw)

    async def end():
        ended.append(True)

    async def before():
        greeted.append(True)
    brain = DealerBrain(Greeter(), on_say=said.append, level_of=lambda: 0.2, on_silent_end=end, on_before_greeting=before, **settings)

    async def push(frame, direction=DOWN):
        pass
    brain.push_frame = push
    brain.stop_report_lag_s = 0.0                            # here "stopped speaking" is reported the moment the agent stops
    return brain, said, ended, greeted


async def start(brain):
    await brain.begin()                                      # what the brain does on the pipeline's StartFrame


async def agent_speaks(brain, secs=T):
    """The agent's line is heard: the output reports it started and, a moment later, stopped."""
    await brain.process_frame(BotStartedSpeakingFrame(), UP)
    await asyncio.sleep(secs)
    await brain.process_frame(BotStoppedSpeakingFrame(), UP)


async def caller_speaks(brain, text, secs=T):
    await brain.process_frame(VADUserStartedSpeakingFrame(), DOWN)
    await asyncio.sleep(secs)
    await brain.process_frame(VADUserStoppedSpeakingFrame(), DOWN)
    await hear(brain, text)


def test_a_caller_who_speaks_first_is_answered_without_a_greeting():
    async def go():
        brain, said, ended, greeted = make()
        await start(brain)
        await asyncio.sleep(T)
        await caller_speaks(brain, "I'm looking for a Toyota.")
        await asyncio.sleep(4 * T)                           # past the greeting's 3
        return brain, said
    brain, said = asyncio.run(go())
    assert said == ["Reply to: I'm looking for a Toyota."] and brain.engine.seen == ["I'm looking for a Toyota."]


def test_a_quiet_start_gets_the_greeting_after_three():
    async def go():
        brain, said, ended, greeted = make()
        await start(brain)
        await asyncio.sleep(2 * T)
        early = list(said)
        await asyncio.sleep(2 * T)
        return early, said, greeted
    early, said, greeted = asyncio.run(go())
    assert early == [] and said == ["Hello, I'm Wazir from AGMC. What kind of car are you after?"] and greeted == [True]


def test_talking_at_the_start_puts_the_greeting_off_until_it_is_quiet_again():
    async def go():
        brain, said, ended, greeted = make()
        await start(brain)
        await brain.process_frame(VADUserStartedSpeakingFrame(), DOWN)
        await asyncio.sleep(5 * T)                           # talking past the 3: no greeting over it
        during = list(said)
        await brain.process_frame(VADUserStoppedSpeakingFrame(), DOWN)       # it gave no turn (the room): greet 3 later
        await asyncio.sleep(4 * T)
        return during, said
    during, said = asyncio.run(go())
    assert during == [] and said == ["Hello, I'm Wazir from AGMC. What kind of car are you after?"]


def test_silence_after_the_agent_gets_are_you_still_there_then_the_goodbye_then_the_end():
    async def go():
        brain, said, ended, greeted = make()
        await start(brain)
        await caller_speaks(brain, "I'm looking for a Toyota.")
        await agent_speaks(brain)
        await asyncio.sleep(7 * T)
        before_8 = list(brain.quiet_prompts)
        await asyncio.sleep(2 * T)
        await agent_speaks(brain)                            # "Hello? Are you still there?" is heard
        await asyncio.sleep(6 * T)
        before_7 = list(brain.quiet_prompts)
        await asyncio.sleep(2 * T)
        await agent_speaks(brain)                            # the goodbye is heard
        await asyncio.sleep(T)
        return brain, before_8, before_7, ended
    brain, before_8, before_7, ended = asyncio.run(go())
    assert before_8 == [] and before_7 == [STILL_THERE_ASK] and brain.quiet_prompts == [STILL_THERE_ASK, SILENT_GOODBYE] and ended == [True]
    assert ("agent", STILL_THERE_ASK) not in brain.session.history          # the conversation goes on from where it was


def test_an_answer_to_are_you_still_there_starts_the_clock_again():
    async def go():
        brain, said, ended, greeted = make()
        await start(brain)
        await caller_speaks(brain, "I'm looking for a Toyota.")
        await agent_speaks(brain)
        await asyncio.sleep(9 * T)
        await agent_speaks(brain)                            # "are you still there?"
        await asyncio.sleep(2 * T)
        await caller_speaks(brain, "Yes, sorry.")
        await agent_speaks(brain)
        await asyncio.sleep(7 * T)                           # 7 after the reply: the goodbye would have come by now; "still there" is 8
        return brain, ended
    brain, ended = asyncio.run(go())
    assert brain.quiet_prompts == [STILL_THERE_ASK] and ended == []


def test_no_clock_while_the_agent_is_talking():
    async def go():
        brain, said, ended, greeted = make()
        await start(brain)
        await caller_speaks(brain, "Tell me about the Camry.")
        await agent_speaks(brain, secs=12 * T)              # a long reply
        await asyncio.sleep(T)
        return brain
    assert asyncio.run(go()).quiet_prompts == []


def test_after_goodbye_from_the_caller_nobody_asks_if_they_are_still_there():
    class Bye(Greeter):
        def handle_stream(self, s, text, **extra):
            yield "Thanks for calling. Goodbye!"
            t = next(super().handle_stream(s, text))       # noqa: F841 (the parent's reply is not used)
            turn = type("T", (), {})()
            turn.debug = type("D", (), {"intent": "end_call", "via": "rules", "draft": False, "risk": "low", "note": ""})()
            turn.ended = True
            yield turn

    async def go():
        brain, said, ended, greeted = make()
        brain.engine = Bye()
        await start(brain)
        await caller_speaks(brain, "That's all, thanks.")
        await agent_speaks(brain)
        await asyncio.sleep(10 * T)
        return brain
    assert asyncio.run(go()).quiet_prompts == []


def test_a_visitor_who_comes_back_hears_welcome_back_if_they_say_nothing():
    async def go():
        brain, said, ended, greeted = make(welcome_back=True)
        brain._speak_opening = False                         # the server: a call that is kept is not greeted again
        await start(brain)
        await asyncio.sleep(4 * T)
        return brain, said
    brain, said = asyncio.run(go())
    assert said == [WELCOME_BACK] and brain.session.history[-1] == ("agent", WELCOME_BACK)


def test_off_greets_at_once_and_never_asks():
    async def go():
        said = []
        brain = DealerBrain(Greeter(), on_say=said.append, level_of=lambda: 0.2)
        pushed = []

        async def push(frame, direction=DOWN):
            pushed.append(frame)
        brain.push_frame = push
        await start(brain)
        await agent_speaks(brain)
        await asyncio.sleep(3 * T)
        return brain, said, pushed
    brain, said, pushed = asyncio.run(go())
    assert said == ["Hello, I'm Wazir from AGMC. What kind of car are you after?"] and brain.quiet_prompts == ["Hello, I'm Wazir from AGMC. What kind of car are you after?"]
    assert [f.text for f in pushed if isinstance(f, TTSSpeakFrame)] == said


def test_the_settings():
    v = load_config(environ={}).voice
    assert (v.greet_if_quiet_secs, v.still_there_secs, v.hang_up_secs, v.calls_kept_secs) == (3.0, 8.0, 7.0, 900.0)


def test_a_turn_taken_while_the_agent_is_still_talking_does_not_start_the_clock_until_it_stops():
    """30 Oct, a test call through a worker: the caller spoke over the agent, the turn was taken while the agent was audible, and "are you still there?" came
    8 s after the caller's turn instead of 8 s after the agent fell silent."""
    async def go():
        brain, said, ended, greeted = make()
        await start(brain)
        await brain.process_frame(BotStartedSpeakingFrame(), UP)
        await caller_speaks(brain, "Do you have any Toyota SUVs?")
        await asyncio.sleep(10 * T)                          # the agent still talking (its reply and more)
        during = list(brain.quiet_prompts)
        await brain.process_frame(BotStoppedSpeakingFrame(), UP)
        await asyncio.sleep(7 * T)
        before_8 = list(brain.quiet_prompts)
        await asyncio.sleep(2 * T)
        return during, before_8, brain
    during, before_8, brain = asyncio.run(go())
    assert during == [] and before_8 == [] and brain.quiet_prompts == [STILL_THERE_ASK]


def test_the_late_stopped_speaking_report_is_allowed_for():
    """The output reports the end of the agent's speech 3 s late (no end-of-sentence marks from our voices): the wait after it is 3 s shorter."""
    async def go():
        brain, said, ended, greeted = make()
        brain.stop_report_lag_s = 3 * T
        await start(brain)
        await caller_speaks(brain, "I'm looking for a Toyota.")
        await agent_speaks(brain)                            # reported 3 late: 5 more of quiet make 8
        await asyncio.sleep(4 * T)
        before = list(brain.quiet_prompts)
        await asyncio.sleep(2 * T)
        return before, brain
    before, brain = asyncio.run(go())
    assert before == [] and brain.quiet_prompts == [STILL_THERE_ASK]
    from dealer_core.voice.brain import DealerBrain as B
    assert B(Greeter()).stop_report_lag_s == 3.0
