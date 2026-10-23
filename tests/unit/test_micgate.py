"""Tap-to-talk: the microphone gate, the end-of-turn notifier, and the protocol message."""
import asyncio

from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InputAudioRawFrame, InputTransportMessageFrame,
                                   VADUserStoppedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection

from dealer_core.server.protocol import PortalSerializer
from dealer_core.voice.micgate import SILENCE_FRAMES, MicGate, StopNotifier


def audio():
    return InputAudioRawFrame(audio=b"\x01\x00" * 320, sample_rate=16000, num_channels=1)


def drive(gate, items):
    out, interruptions = [], []

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        out.append(frame)

    async def interrupt():
        interruptions.append(1)

    gate.push_frame, gate.broadcast_interruption = push, interrupt

    async def go():
        for f, d in items:
            await gate.process_frame(f, d)
    asyncio.run(go())
    return out, interruptions


def mic(on):
    return (InputTransportMessageFrame(message={"type": "mic", "on": on}), FrameDirection.DOWNSTREAM)


def test_audio_passes_while_open_and_is_dropped_while_closed():
    g = MicGate()
    g.close_after_turn()
    out, _ = drive(g, [(audio(), FrameDirection.DOWNSTREAM), (audio(), FrameDirection.DOWNSTREAM)])
    assert out == []
    out, _ = drive(g, [mic(True), (audio(), FrameDirection.DOWNSTREAM)])
    assert len(out) == 1 and isinstance(out[0], InputAudioRawFrame)


def test_tapping_to_finish_feeds_silence_so_the_detector_can_end_the_turn():
    out, _ = drive(MicGate(), [(audio(), FrameDirection.DOWNSTREAM), mic(False), (audio(), FrameDirection.DOWNSTREAM)])
    silent = [f for f in out if isinstance(f, InputAudioRawFrame) and f.audio == bytes(640)]
    assert len(silent) == SILENCE_FRAMES and len([f for f in out if isinstance(f, InputAudioRawFrame)]) == SILENCE_FRAMES + 1     # the later speech is dropped


def test_a_tap_while_the_agent_speaks_interrupts_it_and_a_tap_when_it_is_quiet_does_not():
    g = MicGate()
    g.close_after_turn()
    _, interruptions = drive(g, [(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM), mic(True)])
    assert interruptions == [1] and g.open
    g2 = MicGate()
    g2.close_after_turn()
    _, interruptions = drive(g2, [(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM), (BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM), mic(True)])
    assert interruptions == []


def test_the_notifier_fires_when_the_turn_is_over():
    seen = []

    async def cb():
        seen.append(1)
    n = StopNotifier(cb)

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        pass
    n.push_frame = push

    async def go():
        await n.process_frame(audio(), FrameDirection.DOWNSTREAM)
        await n.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2), FrameDirection.DOWNSTREAM)
    asyncio.run(go())
    assert seen == [1]


def test_the_mic_message_from_the_page_becomes_a_frame_and_other_text_does_not():
    ser = PortalSerializer()
    f = asyncio.run(ser.deserialize('{"type": "mic", "on": true}'))
    assert isinstance(f, InputTransportMessageFrame) and f.message == {"type": "mic", "on": True}
    assert asyncio.run(ser.deserialize('{"type": "hello"}')) is None and asyncio.run(ser.deserialize("not json")) is None
