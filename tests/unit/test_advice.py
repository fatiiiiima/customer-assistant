"""Suggestions for a customer who describes a need (agent/advice.py and the engine's `advise` tool, 5 Nov)."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from dealer_core.agent import advice as AD
from dealer_core.agent.engine import AgentEngine, DECIDE_SYSTEM, ONE_CALL_OTHERS, TOOLS
from dealer_core.app import ROOT, build_app
from dealer_core.config import load_config
from dealer_core.models import CarStatus, CarView
from dealer_core.site.details import CarDetails


def car(i, make, model, price, km=40000, year=2021, body="suv", fuel="petrol"):
    return CarView(stock_id=f"AG-{i}", make=make, model=model, year=year, km=km, price=price, status=CarStatus.AVAILABLE, branch_id="b", body_type=body, fuel=fuel, transmission="automatic",
                   color="white", regional_spec=None, service_history=True, image_urls=())


CARS = [car(1, "Kia", "Rio", 39000, body="hatchback"), car(2, "Toyota", "Land Cruiser", 189000), car(3, "Toyota", "RAV4", 98500), car(4, "Porsche", "911", 495000, body="coupe", year=2022),
        car(5, "Toyota", "Prius", 70000, fuel="hybrid", body="sedan"), car(6, "BMW", "X7", 350000, year=2022), car(7, "Nissan", "Altima", 52000, body="sedan"),
        car(8, "Toyota", "Land Cruiser", 160000, km=90000)]
DET = {"AG-1": CarDetails("AG-1", seats=5, power_hp=84, boot_litres=325, consumption="5.0 L/100km"), "AG-2": CarDetails("AG-2", seats=7, power_hp=304, boot_litres=550, market_price=195000),
       "AG-3": CarDetails("AG-3", seats=5, power_hp=203, boot_litres=580, consumption="7.1 L/100km"), "AG-4": CarDetails("AG-4", seats=4, power_hp=450, accel_0_100=3.5),
       "AG-5": CarDetails("AG-5", seats=5, power_hp=122, boot_litres=370, consumption="4.3 L/100km"), "AG-6": CarDetails("AG-6", seats=7, power_hp=380, boot_litres=450),
       "AG-7": CarDetails("AG-7", seats=5, power_hp=180), "AG-8": CarDetails("AG-8", seats=7, power_hp=304, boot_litres=550)}


def ids(cars):
    return [c.stock_id for c in cars]


def test_the_need_is_read_from_the_callers_words_when_the_model_gave_none():
    assert AD.need_from_words("something for the family with kids") == "family"
    assert AD.need_from_words("i want something sporty and fast") == "sporty"
    assert AD.need_from_words("good on fuel") == "economy"
    assert AD.need_from_words("a long road trip") == "trip"
    assert AD.need_from_words("i am looking for a car") == "any"


def test_a_family_gets_the_cars_with_the_most_seats_then_the_biggest_boot_one_per_model():
    got = AD.shortlist(CARS, DET, "family", 3)
    assert ids(got) == ["AG-2", "AG-6", "AG-3"]                                          # 7 seats (boot 550, 450), then 5 seats; the second Land Cruiser is the same model
    assert all(DET[c.stock_id].seats >= 5 for c in got)


def test_cars_without_the_figures_a_need_depends_on_are_left_out_not_guessed():
    bare = {"AG-2": DET["AG-2"]}
    assert ids(AD.shortlist(CARS, bare, "family", 3)) == ["AG-2"]                         # the others have no recorded seats
    assert AD.shortlist(CARS, {}, "economy", 3)[0].stock_id == "AG-5"                    # the hybrid is known from the stock itself
    assert AD.shortlist(CARS, {}, "family", 3) == []


def test_sporty_economy_luxury_and_budget_each_follow_their_own_figure():
    assert AD.shortlist(CARS, DET, "sporty", 2)[0].stock_id == "AG-4"                    # 450 hp, 3.5 s
    assert AD.shortlist(CARS, DET, "economy", 2)[0].stock_id == "AG-5"                   # hybrid first, then the lowest litres per 100 km
    assert ids(AD.shortlist(CARS, DET, "economy", 3))[1] == "AG-1"
    assert AD.shortlist(CARS, DET, "luxury", 1)[0].stock_id == "AG-4"                    # the dearest of the newer sedans, SUVs and coupes
    assert ids(AD.shortlist(CARS, DET, "budget", 2)) == ["AG-1", "AG-7"]


def test_a_trip_prefers_fewer_kilometres_and_newer_among_the_roomy_cars():
    assert AD.shortlist(CARS, DET, "trip", 1)[0].stock_id in ("AG-6", "AG-2")
    assert "AG-8" not in ids(AD.shortlist(CARS, DET, "trip", 3))                           # 90,000 km and the same model as a better one


def test_no_need_yet_gives_a_spread_of_different_kinds_of_car():
    got = AD.shortlist(CARS, DET, "any", 3)
    assert ids(got) == ["AG-1", "AG-2", "AG-4"] and len({(c.make, c.model) for c in got}) == 3


def test_the_facts_hold_only_recorded_figures_and_forbid_praise_and_unrecorded_claims():
    line = AD.car_line(CARS[1], DET["AG-2"], ", 7 seats, boot 550 litres")
    assert "189,000 dirhams" in line and "40,000 kilometres" in line and "7 seats" in line and "6,000 dirhams under the market reference price" in line
    assert "under the market" not in AD.car_line(CARS[2], DET["AG-3"], "")
    facts = " ".join(AD.facts_for("family", "something for the family", [line]))
    assert "ONE short question" in facts and "at most TWO" in facts and "best seller" in facts and "not listed" in facts
    first = " ".join(AD.facts_for("any", "i am looking for a car", [line, "- second"]))
    assert "Do NOT list cars yet" in first and "at most ONE example" in first and "- second" not in first


def test_the_decision_prompt_and_tool_list_know_advise():
    assert "advise" in TOOLS and "\nadvise " in DECIDE_SYSTEM and "family, sporty, economy" in DECIDE_SYSTEM and "advise <family|sporty" in ONE_CALL_OTHERS


SITE = ROOT.parent / "bmw-portal" / "site" / "data.js"
NOW = datetime(2030, 3, 4, 10, 0, tzinfo=ZoneInfo("Asia/Dubai"))


class Scripted:
    name = "scripted"

    def __init__(self, decisions):
        self.decisions = list(decisions)

    def lead_in(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
        return self.decisions.pop(0) if "TOOL:" in user else "ok"

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
        yield "Here you go. Anything else?"

    def classify(self, *a, **k):
        raise NotImplementedError


@pytest.mark.skipif(not SITE.exists(), reason="the website is not on this machine")
def test_the_engine_suggests_from_the_stock_and_remembers_what_it_suggested():
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=SITE, classifier=Scripted([]), reply_mode="conversational", clock=lambda: NOW.astimezone(timezone.utc),
                    lead_backend=Scripted(["advise family"]))
    e = AgentEngine(app.service, Scripted(["advise family"]), app.engine.reply_writer)
    s = e.new_session()
    out = e._t_advise(s, {"need": "family"}, "something for the family", type("D", (), {"note": ""})())
    assert 1 <= len(s.shown) <= 3 and s.state.value == "answering"
    assert all(c.status == CarStatus.AVAILABLE for c in s.shown)
    text = " ".join(out["facts"])
    assert "These cars in stock fit it best" in text and "seats" in text and "ONE short question" in text
    vague = e._t_advise(e.new_session(), {"need": ""}, "i am looking for a car", type("D", (), {"note": ""})())
    assert "Do NOT list cars yet" in " ".join(vague["facts"]) and vague["awaiting"] is True
    seen = e.new_session()
    seen.filters = {"make": "bmw"}                                                       # they looked at a BMW before: a need is not limited to that make
    other = e._t_advise(seen, {"need": "family"}, "a car for a long road trip with my kids", type("D", (), {"note": ""})())
    assert len(seen.shown) > 1 and any(c.make.lower() != "bmw" for c in seen.shown)
    words = e._t_advise(e.new_session(), {}, "something sporty and fast", type("D", (), {"note": ""})())
    assert "sporty or fast" in " ".join(words["facts"])                                  # no need from the model: read from the words


def test_a_need_the_caller_never_expressed_is_not_trusted_from_the_model():
    assert AD.resolve_need("family", "i am looking for a car", []) == "any"                   # the model read a need into an open request: ask the question instead
    assert AD.resolve_need("family", "something for the family", []) == "family"
    assert AD.resolve_need("", "something sporty", []) == "sporty"                           # no need from the model: the words say it
    assert AD.resolve_need("any", "what do you recommend", ["i want something fast"]) == "sporty"       # an open follow-up keeps the need from a moment ago
    assert AD.resolve_need("luxury", "something for the family", []) == "family"            # the words win over a different need from the model


@pytest.mark.skipif(not SITE.exists(), reason="the website is not on this machine")
def test_an_open_request_prepares_the_suggestion_ahead_but_a_named_car_does_not():
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=SITE, classifier=Scripted([]), reply_mode="conversational", clock=lambda: NOW.astimezone(timezone.utc), lead_backend=Scripted([]))
    e = AgentEngine(app.service, Scripted([]), app.engine.reply_writer)
    keys = lambda text: [k for k, *_ in e._one_call_candidates(e.new_session(), text, more=True)]
    assert "advise" in keys("i am looking for a car") and "advise" in keys("something for the family") and "advise" in keys("what do you recommend")
    assert "advise" not in keys("do you have a bmw x5")
