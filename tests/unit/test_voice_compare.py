"""scripts/pocket_server.py and scripts/score_voices.py: the quiet before a sentence's first sound (docs/ARCH_REVIEW.md section 41: Pocket's "eve" sent audio in
35 ms, but it began with about 0.4 s of quiet)."""
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from pocket_server import RATE, from_first_sound  # noqa: E402
from score_voices import lead_ms  # noqa: E402


def sentence(quiet_s: float) -> np.ndarray:
    return np.concatenate([np.full(int(quiet_s * RATE), 0.001, dtype=np.float32), np.full(RATE // 10, 0.3, dtype=np.float32)])


def test_the_server_drops_the_quiet_before_the_first_sound_but_30_ms():
    x = sentence(0.4)
    out = from_first_sound(x)
    assert len(x) - len(out) == int(0.37 * RATE) and np.all(out[int(0.03 * RATE):] == 0.3)


def test_an_all_quiet_chunk_is_not_sent():
    assert from_first_sound(np.zeros(2400, dtype=np.float32)) is None


def test_a_sentence_that_starts_at_once_is_left_alone():
    x = sentence(0.0)
    assert len(from_first_sound(x)) == len(x)


def test_first_heard_counts_the_quiet_at_the_start(tmp_path):
    p = tmp_path / "s.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((sentence(0.25) * 32767).astype(np.int16).tobytes())
    assert abs(lead_ms(p) - 250) < 1


# ------------------------------------------------------------------------------------------------ faster, shorter pauses (owner, 30 Oct; section 43)
from pocket_server import Faster, PauseCap  # noqa: E402


def streamed(tool, x, step=1000):
    return np.concatenate([tool.feed(x[i:i + step]) for i in range(0, len(x), step)] + [tool.flush()])


def test_faster_is_shorter_at_the_same_pitch():
    t = np.arange(RATE * 3) / RATE
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out = streamed(Faster(1.1), x)
    assert abs(len(out) / len(x) - 1 / 1.1) < 0.02
    mid = out[RATE // 2: -RATE // 2]
    assert abs(np.sum(np.diff(np.sign(mid)) != 0) / len(mid) * RATE / 2 - 220) < 3       # still 220 Hz
    assert abs(float(np.abs(mid).max()) - 0.3) < 0.02


def test_speed_one_leaves_the_audio_alone():
    x = np.random.default_rng(0).normal(0, 0.1, RATE).astype(np.float32)
    assert np.array_equal(streamed(Faster(1.0), x), x)


def test_a_long_pause_is_cut_to_the_cap_and_a_short_one_is_left():
    word = (0.2 * np.sin(2 * np.pi * 200 * np.arange(RATE // 4) / RATE)).astype(np.float32)
    long_gap, short_gap = np.zeros(RATE // 2, dtype=np.float32), np.zeros(int(RATE * 0.08), dtype=np.float32)
    out = streamed(PauseCap(120), np.concatenate([word, long_gap, word, short_gap, word]))
    assert len(out) == 3 * len(word) + int(RATE * 0.12) + len(short_gap)


def test_a_breath_counts_as_a_pause():
    """Quiet means under 5% of the loudest so far, as scripts/score_voices.py counts pauses: a soft breath between words is cut too."""
    word = (0.3 * np.sin(2 * np.pi * 200 * np.arange(RATE // 4) / RATE)).astype(np.float32)
    breath = np.random.default_rng(1).normal(0, 0.009, RATE // 2).astype(np.float32)
    out = streamed(PauseCap(120), np.concatenate([word, breath, word]))
    assert len(out) == 2 * len(word) + int(RATE * 0.12)


# ------------------------------------------------------------------------------------------------ "why does she scream SURE and HI" (owner, 30 Oct; section 44)
from pocket_server import Leveller  # noqa: E402


def loudest(x, w=1200):
    return float(np.sqrt(np.convolve(x * x, np.ones(w) / w, "valid").max()))


def test_a_loud_start_is_turned_down_to_about_the_rest():
    t = np.arange(RATE) / RATE
    loud = (0.45 * np.sin(2 * np.pi * 200 * t[: RATE * 2 // 5])).astype(np.float32)      # "HI" at three times the level of the rest
    rest = (0.15 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    out = streamed(Leveller(0.15, makeup=1.0), np.concatenate([loud, rest]))
    first, after = loudest(out[: len(loud)]), loudest(out[len(loud) + RATE // 4:])
    assert first < 0.75 * loudest(loud) and abs(after - 0.106) < 0.01 and first / after < 2.2   # from 3 times the rest to about 2 (the first frames are caught late)
    assert len(out) == len(loud) + len(rest)


def test_ordinary_speech_is_only_raised_by_the_makeup():
    x = (0.08 * np.sin(2 * np.pi * 200 * np.arange(RATE) / RATE)).astype(np.float32)
    out = streamed(Leveller(0.15, makeup=1.15), x)
    assert np.allclose(out, x * 1.15, atol=1e-4)


def test_the_leveller_holds_back_only_its_look_ahead():
    lev = Leveller(0.15, ahead_ms=20)
    first = lev.feed(np.full(RATE // 10, 0.05, dtype=np.float32))                          # 100 ms in
    assert len(first) == RATE // 10 - RATE // 50                                          # 80 ms out at once; the last 20 ms wait for what follows


# ------------------------------------------------------------------------------------------------ "a little slower, the way they do on Fish Audio" (owner, 30 Oct; section 50)
def test_eve_speaks_at_the_model_s_own_speed_with_pauses_up_to_250_ms():
    from pocket_server import parser

    a = parser().parse_args([])
    assert (a.speed, a.max_pause_ms, a.level) == (1.0, 250, 0.15)


def test_a_comma_pause_is_kept_and_a_long_one_is_still_cut():
    word = (0.2 * np.sin(2 * np.pi * 200 * np.arange(RATE // 4) / RATE)).astype(np.float32)
    comma, long_gap = np.zeros(int(RATE * 0.2), dtype=np.float32), np.zeros(int(RATE * 0.6), dtype=np.float32)
    out = streamed(PauseCap(250), np.concatenate([word, comma, word, long_gap, word]))
    assert len(out) == 3 * len(word) + len(comma) + int(RATE * 0.25)
