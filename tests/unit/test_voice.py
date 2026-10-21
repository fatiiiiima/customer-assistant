"""Voice wiring that needs no microphone, speaker or model: construction, the half-duplex gate and the plain STT/TTS helpers."""
import asyncio

import numpy as np

from dealer_core.voice.stt import hotword_terms
from dealer_core.voice.tts import PiperSherpa, voice_dir


def test_hotwords_skip_comments_and_blanks(tmp_path):
    p = tmp_path / "h.txt"
    p.write_text("# cars\nToyota\n\nLand Cruiser\n", encoding="utf-8")
    assert hotword_terms(p) == ["Toyota", "Land Cruiser"] and hotword_terms(tmp_path / "missing.txt") == []


def test_voice_folder_resolves_the_nested_download_layout(tmp_path):
    (tmp_path / "vits-piper-en_US-joe-medium" / "vits-piper-en_US-joe-medium").mkdir(parents=True)
    assert voice_dir(tmp_path, "joe").name == "vits-piper-en_US-joe-medium" and voice_dir(tmp_path, "joe").parent.name == "vits-piper-en_US-joe-medium"
    assert voice_dir(tmp_path, "some-other-folder").name == "some-other-folder"


def test_pcm_conversion_clips_instead_of_wrapping():
    pcm = np.frombuffer(PiperSherpa.to_pcm16(np.array([0.0, 0.5, 1.5, -1.5], dtype=np.float32)), dtype=np.int16)
    assert list(pcm) == [0, 16383, 32767, -32767]


def test_the_gate_drops_microphone_audio_while_the_agent_speaks_and_a_moment_after():
    from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InputAudioRawFrame

    from dealer_core.voice.gate import HalfDuplexGate

    now = [100.0]
    g = HalfDuplexGate(tail_secs=0.5, clock=lambda: now[0])
    mic = InputAudioRawFrame(audio=b"\0\0" * 320, sample_rate=16000, num_channels=1)
    assert not g.drops(mic)                                         # nobody speaking: the caller is heard
    g.drops(BotStartedSpeakingFrame())
    assert g.drops(mic)                                             # the agent speaks: its own echo is ignored
    g.drops(BotStoppedSpeakingFrame())
    assert g.drops(mic)                                             # and for a moment after, while the room still rings
    now[0] += 0.6
    assert not g.drops(mic)                                         # then the caller is heard again
