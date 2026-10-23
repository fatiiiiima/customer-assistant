"""Is what the caller has said so far a COMPLETE answer to what the agent just asked?

Smart Turn judges the sound of a sentence ending. After a question from the agent, a short answer ("Toyota", "yes", "a white one", "tomorrow at three", a phone number) often
sounds unfinished to it and is held for the whole fallback silence. Judging by length does not work ("Toyota" and "Can I book..." are both under a second), so the words are
checked instead: the endpointer peeks at the speech so far with the speech recognizer, and this function says whether it is an answer to the question just asked.
"""
from __future__ import annotations

from ..dialogue.states import State
from ..nlu import expected, extractors as ex
from ..nlu.datetime_parse import parse_when
from ..nlu.rules import plain_answer

MAX_WORDS = 9


def is_complete_answer(text: str, last_agent: str, state: State, shown: list, vocab: dict, now, tz) -> bool:
    toks = ex.tokens_of(text)
    if not toks:
        return False
    if state is State.CONTACT and ex.contact(text).phone:                  # a whole phone number is long but is one answer
        return True
    if len(toks) > MAX_WORDS:
        return False
    if plain_answer(text) is not None:                                    # yes / no
        return True
    if ex.selection(text, shown):                                         # "the second one", "the Patrol"
        return True
    kind = expected.question_kind(last_agent)
    if kind in expected.NARROWING and (expected.answers_narrowing(kind, text, vocab) or ex.search_filters(text, vocab)):    # "Toyota", "a white one", "any budget", "100,000" (any filter answers any narrowing question)
        return True
    asks_when = state is State.SLOT or kind is None and "what day" in last_agent.lower()
    if (asks_when or "day" in last_agent.lower() or "time" in last_agent.lower()) and parse_when(text, now, tz).found:     # "tomorrow at three", "Saturday"
        return True
    if state is State.CONTACT and (ex.contact(text).phone or ex.contact(text).name):                                          # a phone number or "my name is ..."
        return True
    return False
