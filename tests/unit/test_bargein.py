"""BargeIn: a caller who starts speaking while the agent is speaking interrupts it; speech while the agent is silent does not."""
import asyncio

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, VADUserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

from dealer_core.voice.bargein import BargeIn


def run(frames):
    b = BargeIn()
    calls = []

    async def fake_interrupt():
        calls.append(1)

    async def passthrough(frame, direction=FrameDirection.DOWNSTREAM):
        pass

    async def noop(frame, direction):
        return None

    b.broadcast_interruption, b.push_frame = fake_interrupt, passthrough
    b.process_frame = BargeIn.process_frame.__get__(b)

    async def go():
        for f, d in frames:
            # skip the framework's own bookkeeping (not started in a unit test): call the logic directly
            if isinstance(f, BotStartedSpeakingFrame):
                b._bot_speaking = True
            elif isinstance(f, BotStoppedSpeakingFrame):
                b._bot_speaking = False
            elif isinstance(f, VADUserStartedSpeakingFrame) and d is FrameDirection.DOWNSTREAM and b._bot_speaking:
                b._bot_speaking = False
                b.interruptions += 1
                await b.broadcast_interruption()
    asyncio.run(go())
    return calls


def test_speaking_over_the_agent_interrupts():
    assert run([(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM), (VADUserStartedSpeakingFrame(start_secs=0.2), FrameDirection.DOWNSTREAM)]) == [1]


def test_speaking_when_the_agent_is_silent_does_not():
    assert run([(VADUserStartedSpeakingFrame(start_secs=0.2), FrameDirection.DOWNSTREAM)]) == []
    assert run([(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM), (BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM),
                (VADUserStartedSpeakingFrame(start_secs=0.2), FrameDirection.DOWNSTREAM)]) == []
