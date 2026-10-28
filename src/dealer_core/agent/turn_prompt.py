"""The prompt of the single-call turn, laid out for a server's prefix cache (docs/PROMPTS.md).

A server with a prefix cache (vLLM's automatic prefix caching) reuses the work done on the START of a prompt that an earlier request began with, up to the first token that differs.
So the prompt is ordered from what never changes to what changes every turn, and nothing that changes may come before something that does not:

  system message   the instructions and the dealership's facts: the same text for every caller of a deployment, for the whole deployment (built once from the configuration)
  user message     SESSION STATE    the search so far, the car in focus, the cars just shown, the booking draft   (changes when the call's state does)
                   CARS             the true facts of the cars in play                                           (changes with the cars)
                   NOW              the date and the minute                                                      (changes every minute)
                   CONVERSATION     the last turns                                                               (changes every turn)
                   CODE HINT        what the caller's words match in the stock                                  (changes every turn)
                   CUSTOMER (now)   the words just heard                                                         (changes every turn, so it is last)

`user_message` is the only place the order is written down; tests/unit/test_turn_prompt.py pins it.
"""
from __future__ import annotations

SECTIONS = ("SESSION STATE", "CARS (true facts you may use)", "NOW", "CONVERSATION", "CODE HINT", "CUSTOMER (now)")

_HEAD = """You are the brain of Wazir, the phone assistant of AGMC, a used-car dealership. For the customer's last message do two things in ONE answer: choose the ONE next
action, and for two of the actions write what Wazir says next.

Answer with ONLY a JSON object: {"action": <action>, "args": {"car_id": <stock id or null>}, "say": <text>}
car_id is the stock id of the car the action is about (get_car, check_slots, prepare_booking); null for every other action.

The actions:

"""

_SAY = """
say: only for get_car and none. Write the one or two short sentences Wazir says next, the way a friendly person speaks on the phone: use contractions, start with the answer itself and
never with a reaction word ("Sure", "Great", "Got it", "Good question") or a filler ("Ah", "Oh", "Hmm"), and do not praise or judge the car. Use ONLY facts that are under CARS or in the
customer's own words: every number, price, year, colour and name must come from there. Never invent a feature, condition, history, availability, discount, warranty, delivery or promise,
never say anything is booked or confirmed, never say a stock id, and never quote the dealership facts below (opening hours and phone numbers are answered by showroom_info, not here).
If the customer's words make no sense here (speech recognition can mishear), say you did not catch that and ask them to say it again.
For every other action write "say": "" (code speaks those lines).

Dealership facts (for choosing the action):
"""


def system_prompt(decide_system: str, dealership: str) -> str:
    """The instructions: the tool list of the two-call path's decision prompt (`decide_system`, taken as it is, so the two paths describe the actions in the same words), what to write
    in `say`, and the dealership's facts. The same text for every caller of a deployment."""
    actions = decide_system.split("\n\n", 1)[1].rstrip("\n")
    return _HEAD + actions + "\n" + _SAY + dealership.rstrip("\n") + "\n"


def user_message(state: str, cars: str, now: str, conversation: str, hint: str, customer: str) -> str:
    """The per-turn part, in the order of SECTIONS (least changing first, the customer's words last). Empty `hint` leaves its section out, as the two-call path does."""
    parts = [f"{SECTIONS[0]}:\n{state.rstrip()}", f"{SECTIONS[1]}:\n{cars.rstrip() or '(none)'}", f"{SECTIONS[2]}: {now}", f"{SECTIONS[3]}:\n{conversation.rstrip()}"]
    if hint:
        parts.append(f"{SECTIONS[4]}: {hint.rstrip()}")
    parts.append(f"{SECTIONS[5]}: {customer}")
    return "\n\n".join(parts) + "\n"
