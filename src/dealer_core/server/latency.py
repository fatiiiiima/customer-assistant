"""The two messages that feed the page's latency table (ARCH_REVIEW section 88): where the time of one reply went, step by step.

  timing       {"type": "timing", "seconds": 1.6, "turn": 3, "wait": 0.2, "check": 0.05, "hear": 0.12, "reply": 0.75, "voice": 0.14}
               sent when the first sound starts. `seconds` is from the real end of the caller's speech to that sound (the old field, now with the end-of-turn check included).
               wait  the silence the detector waits before it calls the speech over; check  how long Smart Turn then kept the turn open;
               hear  recognition, from the end of the turn to the words; reply  from the words to the agent's first sentence (the model's decision and the first words);
               voice  from that sentence to the first sound. A step the turn did not have is None.
  turn_detail  {"type": "turn_detail", "turn": 3, "decision": 0.51, "ahead": true, "tool": "search_cars", "by_code": false, "words": 21}
               sent when the agent's turn is decided. `decision` is the model's choice of action (inside `reply`); `ahead` says whether the reply for that action was already
               being written when the decision came (the parallel turn: a hit saves about 0.7 s); None when the parallel turn is not on; `by_code` says the code's rules picked the
               action (voice.rules_decide), so no model call was made for the decision. The page joins the two by `turn`.
               `steps` (when the model wrote the reply; ARCH_REVIEW section 96) breaks "reply words" down, in seconds from the start of the turn (the moment the words reached the agent):
               decision, lookup (the tool), sent (the model request went out; negative = written ahead, during the hold), token (the model's first text arrived), first (the first
               words that may be spoken), spoken (the first sentence handed to the voice), trip (the network round trip to the model, an estimate from a request with no key).
"""
from __future__ import annotations


def _s(value) -> float | None:
    return None if value is None else round(float(value), 3)


def timing_message(total: float, parts: dict, check: float, turn: int) -> dict | None:
    """`parts` is the turn clock's (voice/status.py): "waiting for you to finish", "speech recognition", "reply", "voice". None when no words were recognised since the detector's stop:
    the sound that followed (a "Hello? Are you still there?" after a stretch of room noise, 2 Nov: a 7.4 s "reply") answers nobody."""
    if parts.get("speech recognition") is None:
        return None
    check = max(0.0, check or 0.0)
    return {"type": "timing", "seconds": round(total + check, 2), "turn": turn, "wait": _s(parts.get("waiting for you to finish")), "check": _s(check),
            "hear": _s(parts.get("speech recognition")), "reply": _s(parts.get("reply")), "voice": _s(parts.get("voice"))}


def detail_message(result, turn: int) -> dict:
    d = result.debug
    msg = {"type": "turn_detail", "turn": turn, "decision": _s(d.nlu_ms / 1000) if d.nlu_ms else None, "ahead": d.reply_ahead, "tool": d.intent or None,
           "by_code": d.via.endswith(":rules"), "words": len(result.reply.split())}
    if d.writer_steps:                                  # the breakdown of "reply words" (section 96), all in seconds from the start of the turn
        msg["steps"] = {"decision": _s(d.nlu_ms / 1000) if d.nlu_ms else 0.0, "lookup": _s(d.lookup_ms / 1000) if d.lookup_ms is not None else None,
                        "sent": _s(d.writer_steps.get("sent")), "token": _s(d.writer_steps.get("token")), "first": _s(d.writer_steps.get("first")),
                        "spoken": _s(d.first_say_ms / 1000) if d.first_say_ms is not None else None, "trip": _s(d.trip_ms / 1000) if d.trip_ms is not None else None}
    return msg
