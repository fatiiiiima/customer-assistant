from dealer_core.voice.mic_health import MicHealth

CHUNK = b"\x10\x00" * 320            # 20 ms of 16 kHz audio that is not silence
ZERO = bytes(640)


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def stream(health, clock, pieces):
    """pieces: (chunks, seconds of nothing before them, audio)"""
    for n, gap, audio in pieces:
        clock.t += gap
        for _ in range(n):
            clock.t += 0.02
            health.feed(audio)


def test_a_steady_cadence_is_healthy():
    c = Clock(); h = MicHealth(clock=c)
    stream(h, c, [(150, 0, CHUNK)])
    assert not h.damaged() and h.finish_turn() is None


def test_the_microphone_being_closed_between_turns_is_not_a_dropout():
    c = Clock(); h = MicHealth(clock=c)
    stream(h, c, [(50, 0, CHUNK), (50, 2.5, CHUNK)])         # a 2.5 s pause = closed, not damaged
    assert h.finish_turn() is None


def test_the_27_oct_pattern_is_reported_with_its_size():
    """0.28 s of audio, 0.38 s missing, repeated: what the second call's first three turns looked like."""
    c = Clock(); h = MicHealth(clock=c)
    stream(h, c, [(14, 0, CHUNK), (14, 0.38, CHUNK), (14, 0.38, CHUNK)])
    line = h.finish_turn()
    assert line and "2 dropout(s)" in line and "0.7" in line         # ~0.36 s missing twice
    assert h.finish_turn() is None                                    # the next turn starts from zero


def test_a_long_stretch_of_exact_zeros_is_reported_but_a_short_one_is_not():
    c = Clock(); h = MicHealth(clock=c)
    stream(h, c, [(5, 0, ZERO), (50, 0, CHUNK)])                      # 0.1 s of zeros at the start of every turn is normal
    assert h.finish_turn() is None
    stream(h, c, [(40, 0, ZERO), (50, 0, CHUNK)])                     # 0.8 s of zeros is the damaged start seen in Safari
    assert "0.80 s of digital silence" in h.finish_turn()
