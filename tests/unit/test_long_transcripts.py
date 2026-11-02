"""A long sentence is not a loop (2 Nov, the owner's live call: "I'm asking if this Yaris has been owned by other people before." reached the agent as "um").
The recognition done during Smart Turn's hold judged the transcript with an unknown length counted as 0 s, so anything over 65 characters was thrown away."""
import numpy as np

from dealer_core.voice.services import UNCLEAR, clean_transcript
from dealer_core.voice.stt import is_degenerate

LONG = "No, I'm asking if this Yaris has been owned by other people before."


def test_a_long_sentence_with_no_known_length_is_kept():
    assert len(LONG) > 65 and clean_transcript(LONG) == LONG
    assert clean_transcript("Does this one have a full service history, a panoramic roof, ventilated seats and a head-up display, or is that only the other one?").startswith("Does this one")


def test_the_known_length_still_catches_text_the_audio_cannot_hold():
    loop = "yes please " * 40                                           # 440 characters in 1.5 s
    assert is_degenerate(loop, 1.5) and clean_transcript(loop, 1.5) == UNCLEAR
    assert not is_degenerate(LONG, 4.2) and clean_transcript(LONG, 4.2) == LONG


def test_repeats_are_still_loops_whatever_the_length():
    assert is_degenerate("No" + "!" * 400, None) and clean_transcript("No" + "!" * 400) == UNCLEAR
    assert is_degenerate("a" * 30, None)


def test_the_early_recognition_passes_the_length_of_the_audio_it_was_given():
    import inspect

    from dealer_core.voice import pipeline

    assert "clean_transcript(hear.transcribe(pcm, stt.prompt_for())[0], len(pcm) / 32000)" in inspect.getsource(pipeline.build_pipeline)
    assert np.dtype("<i2").itemsize * 16000 == 32000                      # 16-bit samples at 16 kHz: 32,000 bytes a second
