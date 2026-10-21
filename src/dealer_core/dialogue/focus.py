"""What is the customer asking about right now?  One place that decides, with named operations, so it can be tested and traced.

Every bug of the "it did not understand what I meant" kind came from the same question being answered in several places with
different heuristics: when a caller says something about cars, is it
  * a NEW request            ("a red car", "do you have a Land Cruiser")           -> FRESH
  * a REFINEMENT of the last ("nissan" after "a red car", "under 100 thousand")    -> REFINE
  * a SWITCH of subject      ("camry" after "SUVs under 150 thousand")             -> SWITCH
  * a request for OTHERS     ("another car instead", "show me something else")    -> OTHERS / ASK_INSTEAD
  * "stop asking, show me"   ("any budget", "just show me")                        -> LIST_ALL
  * a question about THE CAR in front of them ("is it red", "how many km")         -> handled as a car question, not a search
  * something the search cannot do ("make it automatic")                           -> CANNOT_FILTER
  * pointing at a car that is not on screen ("the second one")                     -> ASK_WHICH
`decide()` is a pure function of (what is being searched now, what the caller said). The dialogue engine applies the decision, stores a
readable description of it in the session, and the debug trace prints it. Nothing here touches the database except through `fits`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from ..nlu import extractors as ex

IDENTITY = ("make", "model", "body_type")                 # what KIND of car
FIND_CUE = re.compile(r"\b(?:do you have|have you got|got any|i want|i need|looking for|show me|do you sell|any other|something else)\b")


class Op(str, Enum):
    REPLAY = "replay"                  # search the stored request exactly as it is (follow-up to "shall I look for a red one?")
    LIST_ALL = "list_all"              # the caller does not want more questions: read out the matches
    ASK_WHAT = "ask_what"              # nothing to search for yet
    ASK_WHICH = "ask_which"            # pointing at a car that is not on screen
    CANNOT_FILTER = "cannot_filter"    # a wish the search cannot filter on
    ASK_INSTEAD = "ask_instead"        # "another car": drop the one discussed and ask what they want
    OTHERS = "others"                  # "other ones": same request without the cars already shown
    FRESH = "fresh"                    # a new request replaces the old one
    REFINE = "refine"                  # the old request plus this
    SWITCH = "switch"                  # the new thing does not fit the old request: the caller changed subject


@dataclass(frozen=True)
class Cues:
    """What the caller's words carry. Built by the engine from the extractors; `new` is the filters named in this utterance."""

    new: dict
    others: bool = False
    listing: bool = False
    points_at_car: bool = False
    bare: bool = False                 # a few words, no request wording ("nissan", "toyota please")
    refers_to_car: bool = False        # "it", "this", "that", "one": the car in focus
    force: bool = False


@dataclass(frozen=True)
class Ctx:
    spec: dict                         # the request being worked on now
    narrow_asks: int                   # narrowing questions asked and not yet answered
    has_shown: bool                    # a car is in focus or cars were just read out


@dataclass(frozen=True)
class Decision:
    op: Op
    spec: dict                         # the request after this utterance
    reason: str                        # readable, shown in the debug trace
    reset_excluded: bool = False       # forget the cars already shown
    exclude_shown: bool = False        # leave out the cars already shown
    mark_discussed: bool = False       # the car in focus and the cars shown count as shown
    stop_narrowing: bool = False


def decide(ctx: Ctx, cues: Cues, fits: Callable[[dict], bool]) -> Decision:
    """`fits(spec)` says whether anything in stock matches `spec` (used only to tell a refinement from a change of subject)."""
    spec, new = dict(ctx.spec), dict(cues.new)
    if cues.force:
        return Decision(Op.REPLAY, spec, "searching the stored request as it is")
    if cues.listing and spec and ctx.narrow_asks > 0:
        return Decision(Op.LIST_ALL, spec, "no preference given: read out the matches instead of asking again", stop_narrowing=True)
    if not new and not cues.others:
        if spec and cues.points_at_car:
            return Decision(Op.ASK_WHICH, spec, "points at a car that is not on screen")
        if spec:
            return Decision(Op.CANNOT_FILTER, spec, "asks for something the search cannot filter on")
        return Decision(Op.ASK_WHAT, spec, "nothing to search for yet")

    exclude_shown = cues.others
    if cues.others and not new and ctx.has_shown:
        spec.pop("make", None)
        spec.pop("model", None)
        if not spec:
            return Decision(Op.ASK_INSTEAD, spec, "wants a different car than the one discussed, no other preference", reset_excluded=False,
                            mark_discussed=True)
        return Decision(Op.OTHERS, spec, "wants other cars with the same preferences (make and model dropped)", exclude_shown=True,
                        mark_discussed=True)

    has_identity = bool(new.get("make") or new.get("model"))
    names_kind_or_colour = bool(new.get("color") or new.get("body_type"))

    # "a red car" / "an SUV": a request about no car in particular replaces the make, model and body type of the earlier one.
    # (Not when it answers a narrowing question, and not when it says "it"/"this": that is the car in focus.)
    if names_kind_or_colour and not has_identity and ctx.narrow_asks == 0 and not cues.refers_to_car:
        dropped = [k for k in IDENTITY if k in spec]
        for k in IDENTITY:
            spec.pop(k, None)
        spec.update(new)
        why = "a request about no car in particular" + (f": earlier {', '.join(dropped)} no longer applies" if dropped else "")
        return Decision(Op.FRESH if dropped else Op.REFINE, spec, why, exclude_shown=exclude_shown)

    # a bare make/model after a search ("nissan" after "a red car"): it narrows what was asked, it does not start over
    if cues.bare and ctx.spec and has_identity and not names_kind_or_colour:
        base = dict(ctx.spec)
        if new.get("make") and not new.get("model"):
            base.pop("model", None)                           # the old model belonged to the old make
        merged = {**base, **new}
        if fits(merged) or merged.get("color"):
            return Decision(Op.REFINE, merged, "a make or model on its own narrows the current request", exclude_shown=exclude_shown)
        return Decision(Op.SWITCH, dict(new), "does not fit the current request: the caller changed subject", reset_excluded=True,
                        exclude_shown=exclude_shown)

    # a named model, or a make said out of the blue: a new request
    if new.get("model") or (new.get("make") and ctx.narrow_asks == 0):
        return Decision(Op.FRESH, dict(new), "a named car starts a new request", reset_excluded=True, exclude_shown=exclude_shown)

    spec.update(new)
    return Decision(Op.REFINE, spec, "adds to the current request", exclude_shown=exclude_shown)


def about_the_car_in_focus(text: str, has_car: bool) -> bool:
    """A question about THE car on screen ("what colour is it", "is it red", "how many km"), as opposed to a search for cars.
    Price and availability are handled elsewhere; wording that asks to FIND cars ("do you have", "I want") is never a car question."""
    if not has_car or FIND_CUE.search(" ".join(ex.tokens_of(text))):
        return False
    return bool(ex.claims_in_question(text) or set(ex.attributes_asked(text)) & ex.CAR_FACTS)


def describe(spec: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in spec.items()) or "nothing yet"
