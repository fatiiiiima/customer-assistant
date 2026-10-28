"""The answer of the single-call turn (`dialogue.single_call`): its schema, defined here and nowhere else, and the check every answer goes through.

One request returns one JSON object:

    {"action": "get_car", "args": {"car_id": "AG-101"}, "say": "It has 91,000 kilometres."}

  action   one of the agent's tools (agent/engine.py: TOOLS), the same names the two-call path's decision uses
  args     only `car_id`, and only a car that is in play this turn (the car in focus, the cars just shown, the cars the caller's words match) or null. Everything else a tool needs
           (filters, name, phone, day and time) is still read from the caller's own words by code (`AgentEngine._args_for`): a model-written filter cannot override the words.
  say      what the assistant says next, for the actions in SAY_ACTIONS only; "" for every other action (code speaks those, and the exact booking lines stay code-written)

A `say` is NOT trusted when it arrives. It was written before the tool ran, from the facts the prompt carried, so it goes through the reply writer's own sentence checks against what the
tool actually returned (`ReplyWriter.stream(..., say=...)`); a sentence that fails is not spoken and the two-call path's writer is asked instead.

An answer that is not valid (not JSON, not an object, an unknown action, a car that is not in play, a `say` that is too long) raises MalformedTurn. The engine treats that like a model that did
not answer: it asks the two-call path's decision instead (agent/engine.py: `_decide_single`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

SAY_ACTIONS = ("get_car", "none")        # the actions whose result the model has, before the tool runs, every fact of: the cars in play and the conversation
MAX_SAY_CHARS = 320                      # about 45 words; the voice profile asks for at most 30


class MalformedTurn(ValueError):
    """The model's answer is not a valid turn."""


@dataclass(frozen=True)
class Turn:
    action: str
    car_id: str | None
    say: str


def turn_schema(actions: tuple[str, ...] | list[str], car_ids: list[str]) -> dict:
    """The JSON schema of the answer, for the actions and cars of this turn. Property order is the order the model writes them in: the action first, the reply last."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "args", "say"],
        "properties": {
            "action": {"type": "string", "enum": list(actions)},
            "args": {
                "type": "object",
                "additionalProperties": False,
                "required": ["car_id"],
                "properties": {"car_id": {"enum": [*car_ids, None]}},
            },
            "say": {"type": "string", "maxLength": MAX_SAY_CHARS},
        },
    }


def parse_turn(raw: str, actions: tuple[str, ...] | list[str], car_ids: list[str]) -> Turn:
    """The turn in `raw`, or MalformedTurn saying what is wrong with it."""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    try:
        data = json.loads(text)
    except ValueError as e:
        raise MalformedTurn(f"not JSON ({e})") from e
    if not isinstance(data, dict):
        raise MalformedTurn("not a JSON object")
    action = data.get("action")
    if action not in actions:
        raise MalformedTurn(f"unknown action {action!r}")
    args = data.get("args", {})
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise MalformedTurn("args is not an object")
    car = args.get("car_id")
    if car in ("", "null", "none"):
        car = None
    if car is not None and car not in car_ids:
        raise MalformedTurn(f"car {car!r} is not one of the cars in play")
    say = data.get("say", "")
    if say is None:
        say = ""
    if not isinstance(say, str):
        raise MalformedTurn("say is not text")
    say = " ".join(say.split())
    if len(say) > MAX_SAY_CHARS:
        raise MalformedTurn(f"say is {len(say)} characters (at most {MAX_SAY_CHARS})")
    if action not in SAY_ACTIONS:
        say = ""                                                    # code speaks these; whatever the model wrote is not used
    elif say and not say.endswith((".", "!", "?")):
        say += "."                                                  # the JSON string is complete; a reply with no full stop is not a reply cut off
    return Turn(action, car, say)


def same_action(a: tuple[str, str | None], b: tuple[str, str | None], default_car: str | None = None) -> bool:
    """Do two decisions (action, car) choose the same action? The car only matters for the actions that act on one car, and a decision that names none means the car in focus
    (`default_car`): that is what the engine does with it (`AgentEngine._car_arg`)."""
    if a[0] != b[0]:
        return False
    return a[0] not in ("get_car", "check_slots", "prepare_booking") or (a[1] or default_car or None) == (b[1] or default_car or None)
