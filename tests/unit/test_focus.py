"""The interpretation table: what a caller's words mean given what is being searched. One row per past misunderstanding."""
import pytest

from dealer_core.dialogue.focus import Ctx, Cues, Op, about_the_car_in_focus, decide
from dealer_core.nlu import extractors as ex

VOCAB = {"toyota": ["camry", "land cruiser"], "nissan": ["altima", "patrol"], "ford": ["mustang"]}


def cues(text, force=False, **over):
    toks = ex.tokens_of(text)
    c = dict(new=ex.search_filters(text, VOCAB), others=ex.wants_others(text), listing=ex.wants_listing(text),
             points_at_car=ex.points_at_a_car(text), bare=ex.is_bare_vehicle(text), refers_to_car=bool(set(toks) & {"it", "this", "that", "one"}),
             force=force)
    c.update(over)
    return Cues(**c)


def run(text, spec=None, narrow_asks=0, has_shown=False, fits=lambda s: True):
    return decide(Ctx(dict(spec or {}), narrow_asks, has_shown), cues(text), fits)


# (what was being searched, what the caller says) -> (operation, request afterwards)
TABLE = [
    # the transcript bugs
    ({"color": "red"}, "nissan", dict(), Op.REFINE, {"color": "red", "make": "nissan"}),                         # "a red car" then "nissan"
    ({"make": "toyota", "model": "camry"}, "a red car", dict(), Op.FRESH, {"color": "red"}),                      # not "red Camrys"
    ({"make": "toyota", "model": "camry"}, "i want a red car", dict(), Op.FRESH, {"color": "red"}),
    ({"body_type": "suv", "max_price": 150000}, "camry", dict(fits=lambda s: "body_type" not in s), Op.SWITCH, {"make": "toyota", "model": "camry"}),
    ({"make": "toyota"}, "camry", dict(), Op.REFINE, {"make": "toyota", "model": "camry"}),
    ({"color": "red", "make": "toyota", "model": "camry"}, "nissan", dict(), Op.REFINE, {"color": "red", "make": "nissan"}),   # old model dropped
    # new requests
    ({"color": "red"}, "do you have a land cruiser", dict(), Op.FRESH, {"make": "toyota", "model": "land cruiser"}),
    ({"body_type": "suv"}, "do you have a bentley", dict(), Op.FRESH, {"make": "bentley"}),                      # a make we do not stock: still a new request
    # refinements
    ({"body_type": "suv"}, "under 100 thousand", dict(), Op.REFINE, {"body_type": "suv", "max_price": 100000}),
    ({"body_type": "suv"}, "toyota please", dict(narrow_asks=1), Op.REFINE, {"body_type": "suv", "make": "toyota"}),
    ({"make": "toyota", "model": "camry"}, "do you have it in red", dict(), Op.REFINE, {"make": "toyota", "model": "camry", "color": "red"}),
    ({"body_type": "suv", "max_price": 150000}, "an suv", dict(narrow_asks=1), Op.REFINE, {"body_type": "suv", "max_price": 150000}),
    # budget survives a change of kind of car
    ({"max_price": 90000, "body_type": "sedan"}, "an suv", dict(), Op.FRESH, {"max_price": 90000, "body_type": "suv"}),
    # others
    ({"make": "toyota", "model": "camry"}, "i want another car instead", dict(has_shown=True), Op.ASK_INSTEAD, {}),
    ({"make": "toyota", "model": "camry", "max_price": 90000}, "show me something different", dict(has_shown=True), Op.OTHERS, {"max_price": 90000}),
    # nothing to apply
    ({"body_type": "suv"}, "make it automatic", dict(), Op.CANNOT_FILTER, {"body_type": "suv"}),
    ({"body_type": "suv"}, "tell me about the second one", dict(), Op.ASK_WHICH, {"body_type": "suv"}),
    ({}, "hmm", dict(), Op.ASK_WHAT, {}),
    # stop asking
    ({"make": "toyota"}, "any budget", dict(narrow_asks=1), Op.LIST_ALL, {"make": "toyota"}),
    ({"make": "toyota"}, "just show me", dict(narrow_asks=2), Op.LIST_ALL, {"make": "toyota"}),
]


@pytest.mark.parametrize("spec,text,kw,op,after", TABLE)
def test_interpretation(spec, text, kw, op, after):
    d = run(text, spec, **kw)
    assert (d.op, d.spec) == (op, after), f"{text!r} on {spec} -> {d.op.value}: {d.reason}"
    assert d.reason


def test_force_replays_the_stored_request_untouched():
    d = decide(Ctx({"color": "red"}, 0, False), cues("", force=True), lambda s: True)
    assert d.op is Op.REPLAY and d.spec == {"color": "red"}


def test_a_switch_forgets_the_cars_already_shown_but_a_refinement_does_not():
    assert run("camry", {"body_type": "suv"}, fits=lambda s: False).reset_excluded
    assert not run("nissan", {"color": "red"}).reset_excluded
    assert run("show me other toyotas", {"make": "toyota"}, has_shown=True).exclude_shown


# questions about THE car on screen are not searches
@pytest.mark.parametrize("text,has_car,expected", [
    ("what color is the camry is it red", True, True),
    ("is it red", True, True),
    ("how many kilometres has it done", True, True),
    ("is it automatic", True, True),
    ("what color is the camry is it red", False, False),      # no car on screen: nothing to ask about
    ("do you have it in red", True, False),                   # asking to FIND one
    ("i want a red car", True, False),
    ("show me something else", True, False),
    ("how much is it", True, False),                          # price is a negotiation, not a car fact
])
def test_questions_about_the_car_in_focus(text, has_car, expected):
    assert about_the_car_in_focus(text, has_car) is expected
