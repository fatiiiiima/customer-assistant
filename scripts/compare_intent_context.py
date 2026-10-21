r"""Does telling the intent call what the agent just said help or hurt?  Same invented utterances, with and without that context.

  .venv\Scripts\python.exe scripts\compare_intent_context.py
Not a tuned or held-out set. Prints every utterance where the two disagree, and flags the dangerous labels (angry, request_human).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dealer_core.config import load_config  # noqa: E402
from dealer_core.dialogue.states import State  # noqa: E402
from dealer_core.nlu.chain import BackendChain  # noqa: E402
from dealer_core.nlu.consistency import repair  # noqa: E402

Q = {
    State.ANSWERING: "I have a 2019 Nissan Altima at 52,500 dirhams and a 2018 Ford Mustang at 120,000 dirhams. Which one would you like to hear more about?",
    State.NEGOTIATING: "The best I can do right now is 52,200 dirhams, subject to confirmation. Can you work with that?",
    State.INTEREST: "I have 6 SUVs. Do you prefer a particular make, or a maximum mileage?",
    State.SLOT: "I can do Saturday at 12:00 PM or 2:30 PM. Which would you like?",
}
CASES = [
    (State.ANSWERING, "its so hot today"), (State.ANSWERING, "i like football"), (State.ANSWERING, "the weather is nice"), (State.ANSWERING, "yes"),
    (State.ANSWERING, "the first one"), (State.ANSWERING, "tell me more"), (State.ANSWERING, "hmm ok"), (State.ANSWERING, "thanks"),
    (State.ANSWERING, "this is useless"), (State.ANSWERING, "i want to speak to a person"), (State.ANSWERING, "ok whatever"),
    (State.ANSWERING, "can you do 100,000"), (State.ANSWERING, "what is the capital of france"), (State.ANSWERING, "i cant afford that"),
    (State.NEGOTIATING, "yes that works"), (State.NEGOTIATING, "no that is too high"), (State.NEGOTIATING, "its so hot today"),
    (State.NEGOTIATING, "hmm let me think"), (State.NEGOTIATING, "ok fine"), (State.NEGOTIATING, "you are wasting my time"),
    (State.INTEREST, "any budget"), (State.INTEREST, "toyota"), (State.INTEREST, "i dont mind"), (State.INTEREST, "its so hot today"),
    (State.INTEREST, "do you do financing"), (State.INTEREST, "yes"), (State.INTEREST, "thanks"),
    (State.SLOT, "saturday"), (State.SLOT, "4 pm is good"), (State.SLOT, "its so hot today"), (State.SLOT, "no not saturday"),
]


def main() -> int:
    cfg = load_config()
    chain = BackendChain.from_config(cfg, primary="openvino_gpu")
    chain.start()
    diff = bad_without = bad_with = 0
    for state, text in CASES:
        a = chain.classify(state, "S1", text).intent
        b = chain.classify(state, "S1", text, Q[state]).intent
        a2, b2 = repair(a, state, text)[0], repair(b, state, text)[0]
        danger = lambda i: i.value in ("angry", "request_human")
        bad_without += danger(a2) and "useless" not in text and "person" not in text and "wasting" not in text
        bad_with += danger(b2) and "useless" not in text and "person" not in text and "wasting" not in text
        if a2 != b2:
            diff += 1
            print(f"{state.value:11s} {text!r:42s} without context: {a2.value:16s} with context: {b2.value}")
    print(f"\n{len(CASES)} utterances, {diff} differ. Wrongly 'angry'/'request_human' (after the safeguard): without context {bad_without}, with context {bad_with}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
