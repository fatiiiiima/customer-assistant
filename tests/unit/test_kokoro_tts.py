"""Kokoro's pieces: a short first piece so the first sound comes quickly, then clauses."""
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
