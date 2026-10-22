import numpy as np

from dealer_core.voice.services import MIN_RMS, is_phantom, segment_rms


def test_quiet_noise_is_not_speech_and_speech_is():
    rng = np.random.default_rng(1)
    noise = (rng.normal(0, 0.004, 16000) * 32768).astype(np.int16).tobytes()
    speech = (np.sin(np.linspace(0, 400, 16000)) * 0.1 * 32768).astype(np.int16).tobytes()
    assert segment_rms(noise) < MIN_RMS < segment_rms(speech)


def test_repeated_word_hallucinations_are_dropped_but_real_answers_kept():
    assert is_phantom("Fiat, Fiat, Fiat,") and is_phantom("...")
    assert not is_phantom("Kia") and not is_phantom("yes yes") and not is_phantom("Toyota SUV")
