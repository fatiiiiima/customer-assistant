"""Which quality level each turn gets, from how fast the model is answering right now. One governor is shared by every call on a server.

  full    the model writes the whole reply (from the facts the tools gave; every sentence checked). Most natural, needs a fast model.
  prefix  code says the facts at once, the model adds only the follow-up question. Fast on a slow model.
  code    no model writing at all: the code's own wording. The safety net when the model is overloaded or down.

The governor keeps a moving average of how long the model takes to its first sentence in `full` mode, measured on real turns. A turn gets
`full` while that estimate fits the budget (the part of the ~1 s reply target that is left for the model once speech detection, recognition
and the first audio are counted) and the number of turns using the model at once is below its cap. Otherwise it steps down. While it is
stepped down it still tries `full` now and then (when the model is not busy), so it notices when there is room again.

This is what lets the same code run on any machine: on a fast GPU nearly every turn is `full`; on a laptop or at peak load turns become
`prefix`, and only an overloaded or failing model pushes them to `code`. Nothing waits on a slow model to finish before speaking.

Load shedding (`shed_after_ms`): the levels only change how the reply is WRITTEN; a turn the code cannot route still asks the model which action
to take. When the model server is overwhelmed (a GPU fails, a traffic spike) those decisions queue behind each other and every caller waits.
So once the measured decision time passes `shed_after_ms`, a growing share of turns (up to 90%) skip the model altogether and the code decides
and speaks (`model_allowed`). The model's queue drains; the turns that still go to it keep measuring it, so the share falls back to zero as
soon as it recovers.
"""
from __future__ import annotations

import random
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class GovernorStats:
    turns: dict[str, int] = field(default_factory=lambda: {"full": 0, "prefix": 0, "code": 0})
    full_first_ms: float | None = None          # moving average, full mode, model's first sentence
    decide_ms: float | None = None              # moving average, the model's tool decision (the turns code could not route)
    in_flight: int = 0
    shed: int = 0                               # turns that skipped the model because it was overwhelmed


class QualityGovernor:
    def __init__(self, mode: str = "auto", budget_ms: float = 700.0, max_in_flight_full: int = 0, max_in_flight_model: int = 0,
                 explore_every: int = 8, alpha: float = 0.3, shed_after_ms: float = 0.0, rng: random.Random | None = None) -> None:
        self.mode = mode                         # auto | full | prefix | code
        self.budget_ms = budget_ms
        self.max_full = max_in_flight_full       # 0 = no cap of its own (the latency estimate decides)
        self.max_model = max_in_flight_model     # 0 = no cap; above it, turns are spoken by code
        self.explore_every = explore_every
        self.alpha = alpha
        self.stats = GovernorStats()
        self._lock = threading.Lock()
        self._since_full = 0
        self.shed_after_ms = shed_after_ms        # 0 = never shed
        self._rng = rng or random.Random()

    def choose(self) -> str:
        """The level for the next turn."""
        with self._lock:
            if self.mode != "auto":
                level = self.mode
            elif self.max_model and self.stats.in_flight >= self.max_model:
                level = "code"
            elif self.max_full and self.stats.in_flight >= self.max_full:
                level = "prefix"
            elif self.stats.full_first_ms is None or self.stats.full_first_ms <= self.budget_ms:
                level = "full"
            elif self._since_full >= self.explore_every and self.stats.in_flight == 0:
                level = "full"                   # look again: the model may have room now
            else:
                level = "prefix"
            self._since_full = 0 if level == "full" else self._since_full + 1
            self.stats.turns[level] += 1
            return level

    def record_full(self, first_ms: float) -> None:
        """How long a full-mode turn took to its first sentence (only the model's part)."""
        with self._lock:
            old = self.stats.full_first_ms
            self.stats.full_first_ms = first_ms if old is None else (1 - self.alpha) * old + self.alpha * first_ms

    def has_room(self) -> bool:
        """Can the model take extra work that may be thrown away (a decision made ahead of time)? Not while it is slow or nearly full."""
        with self._lock:
            if self.mode == "code" or (self.max_model and self.stats.in_flight >= max(1, self.max_model // 2)):
                return False
            return self.stats.full_first_ms is None or self.stats.full_first_ms <= 2 * self.budget_ms

    def model_allowed(self) -> bool:
        """May this turn ask the model at all? No for a share of turns that grows with how far the decision time is past `shed_after_ms`
        (twice it: 90% shed). The rest keep measuring the model, so shedding stops by itself when it recovers."""
        with self._lock:
            if not self.shed_after_ms or self.stats.decide_ms is None:
                return True
            share = min(0.9, max(0.0, (self.stats.decide_ms - self.shed_after_ms) / self.shed_after_ms))
            if self._rng.random() < share:
                self.stats.shed += 1
                self.stats.turns["code"] += 1
                return False
            return True

    def record_decide(self, ms: float) -> None:
        """How long the model took to name the tool (said before any reply can start on turns the code could not route)."""
        with self._lock:
            old = self.stats.decide_ms
            self.stats.decide_ms = ms if old is None else (1 - self.alpha) * old + self.alpha * ms

    def expected_wait_ms(self) -> float | None:
        """How long a caller is expected to hear nothing on a turn that needs the model's decision: the decision, plus the model's first
        sentence while full turns fit the budget (on prefix turns the code's facts follow the decision at once). None until measured."""
        with self._lock:
            if self.stats.decide_ms is None:
                return None
            first = self.stats.full_first_ms
            writing = first if (self.mode in ("auto", "full") and first is not None and first <= self.budget_ms) else 0.0
            return self.stats.decide_ms + writing

    @contextmanager
    def using_model(self):
        with self._lock:
            self.stats.in_flight += 1
        try:
            yield
        finally:
            with self._lock:
                self.stats.in_flight -= 1

    def snapshot(self) -> dict:
        with self._lock:
            return {"mode": self.mode, "budget_ms": self.budget_ms, "full_first_ms": round(self.stats.full_first_ms or 0),
                    "decide_ms": round(self.stats.decide_ms or 0), "in_flight": self.stats.in_flight, "shed": self.stats.shed,
                    "turns": dict(self.stats.turns), "at": time.time()}
