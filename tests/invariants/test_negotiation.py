"""Deterministic negotiation invariants (acceptance criterion: 100%, no LLM involved).

The property tests use a seeded RNG over thousands of random cars and offer sequences instead of Hypothesis
(Hypothesis is MPL-licensed; the project allows only MIT/Apache/BSD).
"""
import dataclasses
import random

import pytest

from dealer_core.config import load_config
from dealer_core.models import BodyType, Car, CarStatus
from dealer_core.policy.negotiation import Decision, DecisionKind, NegotiationEngine


def make_car(price=100_000, floor=94_000, stock="S1"):
    return Car(stock_id=stock, vin="V" * 17, make="Toyota", model="Camry", year=2020, km=50_000, price=price,
               floor_price=floor, status=CarStatus.AVAILABLE, branch_id="DXB-01", body_type=BodyType.SEDAN)


@pytest.fixture
def engine():
    return NegotiationEngine(load_config(environ={}).negotiation)


def run(engine, car, events):
    state, out = engine.initial_state(car), []
    for e in events:
        d, state = engine.evaluate(car, e, state)
        out.append(d)
    return out, state


# ---- worked examples (placeholder config: tier 50k-150k = 2% auto-accept, 5% max, 3 steps, 3 rounds) -------------
def test_offer_at_or_above_auto_is_accepted_immediately(engine):
    car = make_car()                                           # list 100000, auto = 98000
    assert engine.auto_price(car) == 98_000
    d, _ = engine.evaluate(car, 98_000, engine.initial_state(car))
    assert d.kind is DecisionKind.ACCEPT and d.amount == 98_000


def test_offer_above_list_is_accepted_at_list(engine):
    car = make_car()
    d, _ = engine.evaluate(car, 120_000, engine.initial_state(car))
    assert d.kind is DecisionKind.ACCEPT and d.amount == 100_000


def test_counter_ladder_steps_down_and_ends_at_auto(engine):
    car = make_car()
    assert engine.ladder(car) == [99_400, 98_700, 98_000]


def test_first_counter_is_a_small_step_not_the_floor(engine):
    car = make_car()
    d, _ = engine.evaluate(car, 80_000, engine.initial_state(car))
    assert d.kind is DecisionKind.COUNTER and d.amount == 99_400 and not d.final


def test_full_haggle_to_a_human(engine):
    car = make_car()
    ds, _ = run(engine, car, [80_000, 85_000, 90_000, 90_500])
    assert [d.kind for d in ds] == [DecisionKind.COUNTER, DecisionKind.COUNTER, DecisionKind.COUNTER, DecisionKind.ESCALATE]
    assert [d.amount for d in ds[:3]] == [99_400, 98_700, 98_000] and ds[2].final
    assert ds[3].reason == "below_floor" and ds[3].amount is None


def test_offer_between_floor_and_auto_in_the_last_round_needs_a_manager(engine):
    car = make_car()
    ds, _ = run(engine, car, [90_000, 92_000, 96_000])         # 96000 >= floor 94000 but < auto 98000
    assert ds[2].kind is DecisionKind.NEEDS_MANAGER and ds[2].amount == 96_000 and ds[2].reason == "price_approval"


def test_offer_reaching_the_auto_price_after_a_counter_is_accepted(engine):
    car = make_car()                                           # auto = 98000
    ds, _ = run(engine, car, [90_000, 98_200])
    assert [d.kind for d in ds] == [DecisionKind.COUNTER, DecisionKind.ACCEPT] and ds[1].amount == 98_200


def test_lowest_price_question_returns_a_rung_not_the_limit(engine):
    car = make_car()
    ds, _ = run(engine, car, [None])
    assert ds[0].kind is DecisionKind.COUNTER and ds[0].amount == 99_400 != engine.auto_price(car)


def test_asking_lowest_price_repeatedly_cannot_extract_more_than_the_ladder(engine):
    car = make_car()
    ds, state = run(engine, car, [None] * 6)
    amounts = [d.amount for d in ds]
    assert amounts[:3] == [99_400, 98_700, 98_000] and amounts[3:] == [98_000] * 3        # restated, no further concession
    assert all(d.kind is DecisionKind.COUNTER for d in ds)


def test_misheard_amounts_are_reconfirmed_and_cost_no_round(engine):
    car = make_car()
    ds, state = run(engine, car, [48, 4_800_000, 0])
    assert all(d.kind is DecisionKind.CLARIFY_AMOUNT for d in ds)
    assert state.rounds_used == 0 and state.last_counter is None


def test_state_for_another_car_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.evaluate(make_car(stock="A"), 90_000, engine.initial_state(make_car(stock="B")))


def test_authority_claims_have_no_input_to_change_limits(engine):
    """There is deliberately no parameter through which a caller claim could reach the engine."""
    import inspect
    assert list(inspect.signature(engine.evaluate).parameters) == ["car", "offer", "state"]


def test_floor_is_not_a_field_of_any_decision():
    assert "floor" not in {f.name for f in dataclasses.fields(Decision)}


# ---- properties over thousands of random negotiations -------------------------------------------------------------
def random_car(rng):
    price = rng.randrange(8_000, 600_000, 500)
    floor = max(1, min(price, int(price * rng.uniform(0.80, 1.0))))
    return make_car(price=price, floor=floor, stock="R")


def random_events(rng, car):
    out = []
    for _ in range(rng.randint(1, 8)):
        r = rng.random()
        if r < 0.15:
            out.append(None)                                                         # "what's your lowest price?"
        elif r < 0.20:
            out.append(rng.choice([0, 7, 48, car.price * 5]))                        # junk / misheard
        else:
            out.append(int(car.price * rng.uniform(0.3, 1.15)))
    return out


@pytest.mark.parametrize("seed", range(5))
def test_never_below_floor_and_monotonic_concessions(engine, seed):
    rng = random.Random(seed)
    for _ in range(2_000):
        car = random_car(rng)
        auto, floor = engine.auto_price(car), car.floor_price
        state, last_counter = engine.initial_state(car), None
        for offer in random_events(rng, car):
            d, state = engine.evaluate(car, offer, state)
            if d.amount is not None and d.kind in (DecisionKind.ACCEPT, DecisionKind.COUNTER):
                assert d.amount >= floor, (car, offer, d)                              # THE invariant: never below the floor
                assert d.amount >= auto or d.kind is DecisionKind.ACCEPT and offer is not None and d.amount == min(offer, car.price)
                assert d.amount <= car.price
            if d.kind is DecisionKind.COUNTER:
                assert d.amount >= auto                                                # counters never go below our own authority
                assert last_counter is None or d.amount <= last_counter               # concessions only go down
                last_counter = d.amount
            if d.kind is DecisionKind.ACCEPT:
                assert offer is None or d.amount == min(offer, car.price)
            if d.kind is DecisionKind.NEEDS_MANAGER:
                assert offer is not None and offer >= floor and d.amount == offer       # only ever the customer's own number
            if d.kind is DecisionKind.ESCALATE:
                assert d.amount is None
            assert state.rounds_used <= engine.cfg.max_rounds + 1


@pytest.mark.parametrize("seed", range(3))
def test_offers_below_floor_never_close_a_deal(engine, seed):
    rng = random.Random(100 + seed)
    for _ in range(2_000):
        car = random_car(rng)
        offer = int(car.floor_price * rng.uniform(0.35, 0.999))
        if engine.is_implausible(car, offer) or offer >= car.floor_price:
            continue
        ds, _ = run(engine, car, [offer] * 6)
        assert not any(d.kind in (DecisionKind.ACCEPT, DecisionKind.NEEDS_MANAGER) for d in ds)
        assert ds[-1].kind is DecisionKind.ESCALATE


def test_first_concession_is_never_the_whole_discount(engine):
    rng = random.Random(9)
    for _ in range(2_000):
        car = random_car(rng)
        auto = engine.auto_price(car)
        if car.price - auto < 600:                     # discount smaller than rounding granularity: nothing to step through
            continue
        d, _ = engine.evaluate(car, int(car.price * 0.5), engine.initial_state(car))
        if d.kind is DecisionKind.COUNTER:
            assert d.amount > auto or engine.cfg.counter_steps == 1
