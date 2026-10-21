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


def test_the_default_voice_is_amy():
    from dealer_core.config import load_config

    cfg = load_config(environ={})
    assert cfg.tts.voice == "amy" and cfg.tts.speaker == 0


def test_agmc_and_friends_are_spelled_out_for_the_voice_only():
    from dealer_core.voice.tts import pronounce

    assert pronounce("Welcome to the AGMC dealership.") == "Welcome to the eigh gee em see dealership."
    assert pronounce("It's GCC spec.") == "It's gee see see spec."
    assert pronounce("I have 6 SUVs and an SUV.") == "I have 6 ess you vees and an ess you vee."
    assert pronounce("Agmc is not changed, nor is a Camry") == "Agmc is not changed, nor is a Camry"       # only the exact written forms


def test_leading_fillers_are_removed_but_real_words_are_not():
    from dealer_core.dialogue.draft import clean_draft, strip_fillers

    assert strip_fillers("Ah, good question.") == "Good question."
    assert strip_fillers("Oh! The Land Cruiser is white.") == "The Land Cruiser is white."
    assert strip_fillers("Hmm, not quite.") == "Not quite."
    assert strip_fillers("Ooh, nice pick.") == "Nice pick."
    assert strip_fillers("Ah!") == ""
    assert strip_fillers("Ohio is a state") == "Ohio is a state" and strip_fillers("Wowzers") == "Wowzers" and strip_fillers("Ahead of you") == "Ahead of you"
    assert clean_draft("SAY: Ah, yes, it is.") == "Yes, it is."


def test_the_transcript_line_goes_to_the_console_and_the_file_only_when_asked(tmp_path, capsys):
    from dealer_core.cli.voice import _write_transcript

    f = open(tmp_path / "t.txt", "w", encoding="utf-8")
    _write_transcript("YOU  ", "hello there", True, f)
    _write_transcript("AGENT", "Welcome to AGMC.", False, f)
    f.close()
    assert "YOU  : hello there" in capsys.readouterr().out and "AGENT: Welcome to AGMC." not in capsys.readouterr().out
    text = (tmp_path / "t.txt").read_text(encoding="utf-8")
    assert "YOU  : hello there" in text and "AGENT: Welcome to AGMC." in text


def test_long_sentences_are_spoken_in_short_clauses_so_the_first_sound_comes_early():
    from dealer_core.voice.tts import speech_chunks

    assert speech_chunks("Got it!") == ["Got it!"]
    assert speech_chunks("It is white and has 62,000 kilometres.") == ["It is white and has 62,000 kilometres."]          # short: left whole
    long = "I have a 2018 Toyota Prado at 105,000 dirhams and a 2021 Toyota Land Cruiser at 148,000 dirhams."
    chunks = speech_chunks(long)
    assert len(chunks) == 2 and chunks[0].endswith("dirhams") and chunks[1].startswith("and a 2021") and " ".join(chunks) == long
    with_commas = "Ah, the 2021 Land Cruiser is white, it has 62,000 kilometres, and it is priced at 148,000 dirhams."
    assert " ".join(speech_chunks(with_commas)) == with_commas and max(len(c.split()) for c in speech_chunks(with_commas)) <= 11


def test_the_turn_clock_reports_total_and_parts_once_per_turn():
    from pipecat.frames.frames import BotStartedSpeakingFrame, TranscriptionFrame, TTSSpeakFrame, VADUserStoppedSpeakingFrame

    from dealer_core.voice.status import TurnClock

    now, got = [10.0], []
    c = TurnClock(0.3, lambda total, parts: got.append((total, parts)), clock=lambda: now[0])
    c.see(BotStartedSpeakingFrame(), "vad")                                   # the greeting: no caller turn yet, nothing to report
    assert got == []
    c.see(VADUserStoppedSpeakingFrame(stop_secs=0.3), "vad")
    now[0] = 10.2
    c.see(TranscriptionFrame("hi", "u", "t"), "stt")
    now[0] = 10.9
    c.see(TTSSpeakFrame("Hello"), "brain")
    now[0] = 11.0
    c.see(BotStartedSpeakingFrame(), "vad")
    total, parts = got[0]
    assert abs(total - 1.3) < 1e-6                                            # 0.3 s of silence waited + 1.0 s until the first sound
    assert abs(parts["speech recognition"] - 0.2) < 1e-6 and abs(parts["reply"] - 0.7) < 1e-6 and abs(parts["voice"] - 0.1) < 1e-6
    c.see(BotStartedSpeakingFrame(), "vad")
    assert len(got) == 1                                                       # only once per turn
