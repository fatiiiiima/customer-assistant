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
