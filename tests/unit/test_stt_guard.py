import numpy as np

from dealer_core.voice.services import MIN_RMS, collapse_repeats, is_phantom, segment_rms


def test_quiet_noise_is_not_speech_and_speech_is():
    rng = np.random.default_rng(1)
    noise = (rng.normal(0, 0.004, 16000) * 32768).astype(np.int16).tobytes()
    speech = (np.sin(np.linspace(0, 400, 16000)) * 0.1 * 32768).astype(np.int16).tobytes()
    assert segment_rms(noise) < MIN_RMS < segment_rms(speech)


def test_repeated_word_hallucinations_are_dropped_but_real_answers_kept():
    assert is_phantom("Fiat, Fiat, Fiat,") and is_phantom("...")
    assert not is_phantom("Kia") and not is_phantom("yes yes") and not is_phantom("Toyota SUV")


def test_a_looped_phrase_is_kept_once():
    assert collapse_repeats("My name is, my name is, my name is, my name is, my name is") == "My name is"
    assert collapse_repeats("I want the white one") == "I want the white one"
    assert collapse_repeats("yes yes") == "yes yes"
    assert collapse_repeats("zero five zero, one two three") == "zero five zero, one two three"
    assert collapse_repeats("zero five five five one two") == "zero five five five one two"


def test_the_quiet_gate_is_a_setting_and_stays_where_quiet_speech_is_still_heard():
    """docs/SCALING.md: at 0.012 the owner's 10%-volume clips went 20/24 -> 0/24 heard, so the gate must stay below that."""
    from dealer_core.config import load_config

    cfg = load_config(environ={}).voice
    assert cfg.min_rms == MIN_RMS < 0.012
    assert load_config(environ={"VA__VOICE__MIN_RMS": "0.01"}).voice.min_rms == 0.01


def test_the_recognizer_gate_follows_its_setting_and_remembers_the_level():
    import asyncio

    from dealer_core.voice.services import OpenVinoWhisperSTTService

    class Engine:
        name = "fake"

        def transcribe(self, pcm, prompt=None):
            return "goodbye", 1.0

    # a steady tone at amplitude 0.014 has a level of about 0.0099
    tone = (np.sin(np.arange(16000) / 5) * 0.014 * 32768).astype(np.int16).tobytes()

    async def words(min_rms):
        stt = OpenVinoWhisperSTTService(engine=Engine(), min_rms=min_rms)
        out = [f.text async for f in stt.run_stt(tone)]
        return out, stt.last_level

    heard, level = asyncio.run(words(0.005))
    assert heard == ["goodbye"] and 0.009 < level < 0.011
    refused, level2 = asyncio.run(words(0.02))
    assert refused == [] and level2 == level            # dropped, but the level is still known
