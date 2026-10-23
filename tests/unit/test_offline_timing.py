"""The per-turn timing reduction: stages are matched in order, so no stage time can be negative and an old reply cannot be mistaken for a new one."""
from dealer_core.voice.offline import reduce_log


def log(*events):
    return [(t, n, k) for t, n, k in events]


def test_a_clean_turn():
    t = reduce_log(log((1.0, "speech", "end:0"), (1.2, "vad", "x"), (1.4, "stt", "x"), (2.0, "brain", "x"), (2.1, "tts", "x")))
    assert round(t[0].total_ms) == 1100 and round(t[0].endpointing_ms) == 200 and round(t[0].brain_ms) == 600


def test_the_previous_replys_later_sentences_and_audio_are_not_this_turns():
    # turn 0's reply keeps coming (brain at 5.5, audio at 5.6) after turn 1's speech ended at 5.0 and its transcript arrived at 6.0
    events = log((1.0, "speech", "end:0"), (1.2, "vad", "x"), (1.4, "stt", "x"), (2.0, "brain", "x"), (2.1, "tts", "x"),
                 (5.0, "speech", "end:1"), (5.5, "brain", "x"), (5.6, "tts", "x"), (5.3, "vad", "x"), (6.0, "stt", "x"), (6.5, "brain", "x"), (6.6, "tts", "x"))
    turns = reduce_log(events)
    assert all(v >= 0 for t in turns for v in (t.endpointing_ms, t.stt_ms, t.brain_ms, t.tts_first_ms, t.total_ms))
    assert round(turns[1].total_ms) == 1600                     # 6.6 - 5.0, not 5.6 - 5.0


def test_a_turn_with_no_reply_is_marked_and_does_not_steal_the_next_one():
    turns = reduce_log(log((1.0, "speech", "end:0"), (1.2, "vad", "x"), (1.4, "stt", "x"), (3.0, "speech", "end:1"), (3.2, "vad", "x"), (3.4, "stt", "x"),
                           (3.9, "brain", "x"), (4.0, "tts", "x")))
    assert turns[0].note == "no reply audio" and turns[0].total_ms == 0 and round(turns[1].total_ms) == 1000
