"""Negotiation policy engine: plain code, no LLM. The LLM never decides or sees a price; it only phrases a Decision.

Model
  list    = car.price                                  (what we advertise)
  auto    = lowest price the AGENT may agree to by itself = list * (1 - tier.auto_accept_pct), never below the floor
  floor   = car.floor_price                            (manager's limit; never spoken, never returned in a Decision)
  ladder  = `counter_steps` rungs from list down to auto; rung k is offered in round k
Rules
  * offer >= auto (or >= a counter we already made)  -> ACCEPT (subject to confirmation)
  * otherwise counter with the rung for this round; concessions only go down and only in steps
  * last round: offer >= floor -> NEEDS_MANAGER (a manager must approve); offer < floor -> one final counter at `auto`
  * after the rounds are used: offer >= floor -> NEEDS_MANAGER, otherwise ESCALATE to a human (price below floor)
  * "what's your lowest price?" is a round like any other and returns the next rung, never the limit
  * implausible amounts (probably misheard) are re-confirmed and do not consume a round
Authority claims ("I'm the manager") are not an input: nothing a caller says changes these limits.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from ..config import NegotiationCfg, TierCfg
from ..models import Car


class DecisionKind(str, Enum):
    ACCEPT = "accept"
    COUNTER = "counter"
    NEEDS_MANAGER = "needs_manager"
    ESCALATE = "escalate"
    CLARIFY_AMOUNT = "clarify_amount"


@dataclass(frozen=True)
class NegotiationState:
    """Per car, per call. Immutable: `evaluate` returns the next state."""

    stock_id: str
    rounds_used: int = 0
    last_counter: int | None = None


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    # The ONLY price a reply may speak. It is the customer's own offer or a counter we chose, never the floor.
    amount: int | None
    round_no: int                   # the round this decision belongs to (1-based)
    rounds_left: int
    final: bool = False             # a COUNTER that is our last word
    reason: str = ""                # e.g. "below_floor", "price_approval", "implausible_amount"


def _round_to(x: float, step: int = 100, up: bool = True) -> int:
    q = x / step
    return int(-(-q // 1) if up else q // 1) * step


class NegotiationEngine:
    def __init__(self, cfg: NegotiationCfg) -> None:
        self.cfg = cfg

    def tier(self, price: int) -> TierCfg:
        for t in self.cfg.tiers:
            if t.up_to is None or price <= t.up_to:
                return t
        raise AssertionError("config validation guarantees a final tier with up_to=None")

    def auto_price(self, car: Car) -> int:
        """Lowest price the agent may agree to without a manager. Never below the floor."""
        t = self.tier(car.price)
        return min(car.price, max(car.floor_price, _round_to(car.price * (1 - t.auto_accept_pct / 100), up=True)))

    def ladder(self, car: Car) -> list[int]:
        """Rungs for rounds 1..counter_steps: from just below list down to `auto`; non-increasing, last == auto."""
        auto, steps = self.auto_price(car), self.cfg.counter_steps
        # Round UP to 100 AED so a rung is never below `auto`; cap at list so rounding cannot push a rung above the price.
        rungs = [min(car.price, _round_to(car.price - (car.price - auto) * k / steps, up=True)) for k in range(1, steps + 1)]
        rungs[-1] = auto
        for i in range(1, len(rungs)):          # concessions only go down
            rungs[i] = min(rungs[i], rungs[i - 1])
        return rungs

    def initial_state(self, car: Car) -> NegotiationState:
        return NegotiationState(stock_id=car.stock_id)

    def is_implausible(self, car: Car, offer: int) -> bool:
        lo, hi = self.cfg.implausible_low_pct / 100, self.cfg.implausible_high_pct / 100
        return offer <= 0 or offer < car.price * lo or offer > car.price * hi

    def evaluate(self, car: Car, offer: int | None, state: NegotiationState) -> tuple[Decision, NegotiationState]:
        """Decide the response to `offer` (AED), or to "what's your lowest price?" when offer is None."""
        if state.stock_id != car.stock_id:
            raise ValueError("negotiation state belongs to a different car")
        max_rounds = self.cfg.max_rounds
        n = state.rounds_used + 1
        left = max(0, max_rounds - n)

        if offer is not None and self.is_implausible(car, offer):
            return Decision(DecisionKind.CLARIFY_AMOUNT, offer if offer > 0 else None, state.rounds_used, max_rounds - state.rounds_used,
                            reason="implausible_amount"), state          # no round consumed, no state change

        auto = self.auto_price(car)
        next_state = replace(state, rounds_used=min(n, max_rounds + 1))    # capped: extra haggling changes nothing

        # 1. Accept anything at or above what the agent may agree to alone. (Every counter we ever made is >= auto,
        #    so an offer that meets an earlier counter is covered by this too.)
        if offer is not None and offer >= auto:
            return Decision(DecisionKind.ACCEPT, min(offer, car.price), n, left), next_state

        # 2. Rounds exhausted: no more concessions.
        if n > max_rounds:
            spent = replace(state, rounds_used=max_rounds + 1)       # capped: more haggling does not change anything
            if offer is not None and offer >= car.floor_price:
                return Decision(DecisionKind.NEEDS_MANAGER, offer, max_rounds + 1, 0, reason="price_approval"), spent
            if offer is None and state.last_counter is not None:
                return Decision(DecisionKind.COUNTER, state.last_counter, max_rounds + 1, 0, final=True), spent   # restate only
            return Decision(DecisionKind.ESCALATE, None, max_rounds + 1, 0, reason="below_floor"), spent

        # 3. Last round: a manager may be able to approve an offer between floor and auto.
        if n == max_rounds and offer is not None and offer >= car.floor_price:
            return Decision(DecisionKind.NEEDS_MANAGER, offer, n, 0, reason="price_approval"), next_state

        # 4. Counter with this round's rung (never above the previous counter, never below auto).
        rung = self.ladder(car)[min(n, self.cfg.counter_steps) - 1]
        if state.last_counter is not None:
            rung = min(rung, state.last_counter)
        rung = max(rung, auto)
        # (An offer >= rung was already accepted in step 1, because rung >= auto and rung <= last_counter.)
        return Decision(DecisionKind.COUNTER, rung, n, left, final=n >= max_rounds), replace(next_state, last_counter=rung)
