"""Kokoro's pieces: a short first piece so the first sound comes quickly, then clauses."""
import numpy as np
from dealer_core.voice.kokoro_tts import first_piece_split, kokoro_runtime


def test_the_first_piece_is_short_and_nothing_is_lost():
    text = "The RAV4 is a 2021 in white, with 33,000 kilometres on it, at 98,500 dirhams."
    pieces = first_piece_split(text)
    assert len(pieces[0].split()) <= 7 and " ".join(pieces) == text
    assert first_piece_split("Want to book a test drive?") == ["Want to book a test drive?"]


def test_an_opener_is_its_own_piece_and_the_next_one_is_short():
    assert first_piece_split("Sure, we've got 4 BMWs in at the moment.") == ["Sure,", "we've got 4 BMWs in at the moment."]
    text = "Right, it's the 2021 Toyota RAV4 in white, with 33,000 kilometres on it, at 98,500 dirhams."
    pieces = first_piece_split(text)
    assert pieces[0] == "Right," and len(pieces[1].split()) <= 8 and " ".join(pieces) == text
    assert first_piece_split("Of course. What day and time suit you?") == ["Of course.", "What day and time suit you?"]
    assert first_piece_split("Okay.") == ["Okay."]


def test_openvino_on_intel_and_amd_onnxruntime_on_apple_chips_unless_chosen():
    assert kokoro_runtime("auto", "AMD64") == "openvino"                  # Windows on Intel / AMD
    assert kokoro_runtime("auto", "x86_64") == "openvino"
    assert kokoro_runtime("auto", "arm64") == "onnxruntime"               # Apple chips: OpenVINO distorts this model's audio
    assert kokoro_runtime("auto", "aarch64") == "onnxruntime"
    assert kokoro_runtime("openvino", "arm64") == "openvino"              # tts.kokoro_runtime set by hand wins
    assert kokoro_runtime("onnxruntime", "AMD64") == "onnxruntime"


# ------------------------------------------------------------------------------------------------ B: openings first (voice.opening_first, owner, 29 Oct)
from dealer_core.voice.kokoro_tts import OPENINGS, KokoroOV, opening_first_split  # noqa: E402


def test_a_sentence_that_starts_with_an_opening_gives_it_as_a_piece_of_its_own():
    assert opening_first_split("There's the Macan at 149,000 dirhams.") == ["There's the", "Macan at 149,000 dirhams."]
    assert opening_first_split("It has a 3.0-litre V6 twin-turbo engine with 340 horsepower.")[0] == "It has"
    assert opening_first_split("The best for a family we have is the Tucson, with 5 seats and a 620-litre boot.")[0] == "The best for a family we have is"


def test_any_other_sentence_starts_with_three_words_or_four_to_a_comma():
    assert opening_first_split("Today I can do 9 in the morning or 2 in the afternoon.")[0] == "Today I can"
    assert opening_first_split("Two previous owners, and a full service history.")[0] == "Two previous owners,"
    assert opening_first_split("Perfect, tomorrow at 2 PM.")[0] == "Perfect,"                     # an opener word stays a piece of its own, as with A
    assert opening_first_split("It's black.") == ["It's black."]


def test_nothing_is_lost_or_reordered():
    for text in ["There's the BMW 320i at 119,000 dirhams, the BMW 530i at 189,000, the BMW X5 at 209,000 and the BMW M3 at 289,000.",
                 "It's the 2019 BMW X5 in white, with 71,000 kilometres on it, at 209,000 dirhams.", "Would you like to come and see it?"]:
        assert " ".join(opening_first_split(text)).split() == text.split()


def test_the_setting_chooses_between_b_and_a_and_b_keeps_its_openings_ready():
    rendered = []

    class Stub(KokoroOV):
        def __init__(self, b):
            self._cache, self._cache_size, self.opening_first, self._pinned = {}, 300, b, {}

        def _say(self, piece):
            rendered.append(piece)
            return piece

    text = "There's the Macan at 149,000 dirhams."
    assert Stub(True).pieces(text)[0] == "There's the" and Stub(False).pieces(text) == first_piece_split(text) == [text]      # A: six words, one piece
    b = Stub(True)
    b.preload([])
    assert set(OPENINGS) <= set(b._pinned) and set(OPENINGS) <= set(rendered)
    a = Stub(False)
    rendered.clear()
    a.preload([])
    assert rendered == [] and a._pinned == {}


# ------------------------------------------------------------------------------------------------ joins (voice.join_silence_ms, owner 29 Oct: "the pauses are awful")
from dealer_core.voice.kokoro_tts import trim_join  # noqa: E402


def _piece(lead_ms, speech_ms, tail_ms, rate=24000):
    n = lambda ms: rate * ms // 1000
    return np.concatenate([np.zeros(n(lead_ms)), np.sin(np.arange(n(speech_ms)) / 5) * 0.3, np.zeros(n(tail_ms))]).astype(np.float32)


def test_the_silence_at_a_join_is_cut_and_the_speech_kept():
    x = _piece(40, 500, 90)
    both = trim_join(x, 24000, True, True, 30)
    assert 500 <= len(both) / 24 <= 500 + 10 + 30 + 10                      # the speech, about 10 ms before it and 30 ms after (to the 10 ms frame)
    assert np.allclose(trim_join(x, 24000, False, False, 30), x)             # the sentence's own start and end are left alone
    assert len(trim_join(x, 24000, False, True, 30)) / 24 >= 540             # its start untouched


def test_only_the_joins_inside_a_sentence_are_trimmed():
    class Stub(KokoroOV):
        def __init__(self):
            self._cache, self._cache_size, self.opening_first, self._pinned, self.join_silence_ms = {}, 300, False, {}, 30
            self.sample_rate = 24000

        def _say(self, piece):
            return _piece(40, 500, 90)

    first, middle, last = list(Stub()._joined(["a", "b", "c"]))
    assert len(first) / 24 > 570 - 1 and len(last) / 24 > 590 - 1          # the sentence starts and ends with its full silence
    assert len(middle) / 24 < 560                                            # in between, both ends cut
    assert len(list(Stub()._joined(["one piece"]))[0]) / 24 >= 629          # a sentence of one piece is not touched
