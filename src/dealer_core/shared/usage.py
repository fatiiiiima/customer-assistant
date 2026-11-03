"""Counters of the work the shared services are asked for, so the capacity of a deployment can be worked out from measured numbers instead of assumptions
(ARCH_REVIEW section 102). Process-wide, thread-safe, numbers only: no transcripts, no phone numbers. A worker shows them at GET /usage; scripts/load_test.py takes the
difference over a run and divides by the turns answered.

  model.post / model.stream      requests to the model: a whole answer (decisions) and a streamed one (replies)
  model.stream_cut               replies stopped before their end (written ahead and not used, or the caller spoke)
  model.error                    requests that failed
  model.input_chars              characters sent to the model (4 or so to a token)
  stt.request / stt.audio_s      clips sent to the recogniser and their seconds of audio; stt.error
  stt.final / stt.final_from_early   the turn's own reading, and one taken from the early start's reading
  spec.start_early / spec.start_hold / spec.skipped / spec.dropped / spec.reply_used   the early start's work
  tts.request / tts.audio_s / tts.cache_hit / tts.fallback / tts.cut   sentences asked of the voice server and the speech made
  turn.answered / turn.sentences / turn.reply_words   turns answered, sentences handed to the voice, words said
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_counts: dict[str, float] = {}


def add(name: str, n: float = 1.0) -> None:
    with _lock:
        _counts[name] = _counts.get(name, 0.0) + n


def snapshot() -> dict[str, float]:
    with _lock:
        return {k: round(v, 3) for k, v in sorted(_counts.items())}


def reset() -> None:
    with _lock:
        _counts.clear()


def delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """What was counted between two snapshots (of one process, or the sum of several)."""
    return {k: round(after.get(k, 0.0) - before.get(k, 0.0), 3) for k in sorted(set(before) | set(after)) if after.get(k, 0.0) != before.get(k, 0.0)}


def per_turn(counts: dict[str, float]) -> dict[str, float]:
    """The counts divided by the turns answered in the same stretch."""
    turns = counts.get("turn.answered", 0.0)
    return {k: round(v / turns, 3) for k, v in counts.items() if turns and k != "turn.answered"}
