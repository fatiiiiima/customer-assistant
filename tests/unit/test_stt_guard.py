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
