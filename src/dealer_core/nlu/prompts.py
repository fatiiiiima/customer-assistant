"""Prompts and JSON-schema grammars for the NLU call.

Two prompt styles (config `prompt_mode`):
  * state  : a short prompt listing only the current state's intents. Best on the iGPU, where a cache miss is cheap.
  * static : ONE prompt listing every intent; the state is enforced by the grammar only. The prefix never changes,
             so the KV cache always hits. Required on CPU, where a cache miss costs 1.2 s+ (Phase 0b).
The grammar is identical in both: the model can only emit an intent that is valid in the current state.
"""
from __future__ import annotations

from ..dialogue.states import INTENT_DOC, SLOT_SCHEMAS, STATE_SPEC, Intent, State

_INTRO = "You are the language-understanding module of a used-car dealership phone agent in the UAE."
_RULES = ("Prices and kilometres are integers in AED and km (fifty five thousand -> 55000). Years like \"twenty twenty one\" "
          "-> 2021. phone is digits only. Keep dates and times as the customer said them.")


def state_system(state: State) -> str:
    intents, _ = STATE_SPEC[state]
    lines = "\n".join(f"{i.value}: {INTENT_DOC[i]}" for i in intents)
    return (f"{_INTRO}\nConversation state: {state.value}. Output ONE JSON object {{\"intent\": ..., \"slots\": {{...}}}}.\n"
            "Only include slots the customer actually said; never invent values. Do not answer the customer.\n"
            f"Intents:\n{lines}\n{_RULES}")


def static_system() -> str:
    lines = "\n".join(f"{i.value}: {INTENT_DOC[i]}" for i in Intent)
    return (f"{_INTRO}\nRead the customer's utterance and output ONE JSON object: {{\"intent\": ..., \"slots\": {{...}}}}.\n"
            "Do not answer the customer. Only include slots the customer actually said. Never invent values.\n"
            f"Intents:\n{lines}\n"
            "Rules: prices and kilometres are integers in AED and km (fifty five thousand -> 55000).\n"
            "Years like \"twenty twenty one\" -> 2021. phone is digits only. Keep dates and times as the customer said them.\n"
            "The STATE line tells you what the agent is doing; pick only an intent that makes sense in that state.")


def system_prompt(state: State, mode: str) -> str:
    return static_system() if mode == "static" else state_system(state)


def user_message(state: State, car: str | None, text: str) -> str:
    return f"STATE: {state.value}\nCAR_IN_FOCUS: {car or 'none'}\nCUSTOMER: {text}"


def intent_only_schema(state: State) -> dict:
    """Grammar for the production call: just the intent (~6 tokens). Slots come from code."""
    intents, _ = STATE_SPEC[state]
    return {"type": "object", "properties": {"intent": {"type": "string", "enum": [i.value for i in intents]}},
            "required": ["intent"]}


def full_schema(state: State) -> dict:
    """Grammar for accuracy comparisons: intent + the slots valid in this state."""
    intents, slots = STATE_SPEC[state]
    return {"type": "object",
            "properties": {"intent": {"type": "string", "enum": [i.value for i in intents]},
                           "slots": {"type": "object", "properties": {k: SLOT_SCHEMAS[k] for k in slots}}},
            "required": ["intent", "slots"]}


OPEN_ANSWER_SYSTEM = ("You are a friendly voice assistant at a used car dealership in the UAE. Answer the caller in at most "
                      "{n} short spoken sentences. No lists, no markdown, no links. Use only the facts given in CONTEXT; if "
                      "the answer is not there, say you will check with a colleague.")
