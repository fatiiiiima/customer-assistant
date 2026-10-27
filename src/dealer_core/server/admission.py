"""How many calls this worker takes, set by how fast it is answering right now, not by a number guessed in advance.

A fixed call cap is tuned on one machine. On a slower machine, or a busier model server, the replies get slow before the cap is reached, and
every caller on the worker waits longer. This controller measures instead: every reply's time (end of the caller's speech -> first sound) is
recorded, and every few seconds the cap is adjusted from the 90th percentile of the recent replies:

  * slower than the target (1 s): the cap drops to 75% of itself (never below `min_calls`). Calls already running are never dropped; new ones
    go to other workers (the router reads this cap) or, when every worker is full, are told "busy" and the page offers text chat.
  * faster than the target (under 90% of it) while the cap is actually what limits the calls: the cap rises by one, up to `max_calls`
    (a hard ceiling for memory).
  * too few recent replies to judge: no change.
  * every cut is checked: if the replies after it are not clearly faster, the slowness does not come from this worker's load (a model that is
    slow even with one caller, a slow network): the cut is undone and no cut is made for a minute (`slow_regardless` in the snapshot, for an
    alert). Turning callers away would cost them the call and save nothing.

This is the scheme networks use to share a link without collapsing it (additive increase, multiplicative decrease): it backs off fast when the
target is missed and probes upward slowly, so it settles just under what the machine and the model server can serve within the target,
whatever they are. After a cut, only replies measured after the cut count, so one slow burst is not punished twice.
"""
from __future__ import annotations

import threading
import time
from collections import deque


class AdmissionController:
    def __init__(self, max_calls: int, target_s: float = 1.0, min_calls: int = 1, start_calls: int | None = None, window_s: float = 30.0,
                 min_samples: int = 12, adjust_every_s: float = 5.0, backoff: float = 0.75, headroom: float = 0.9, hold_s: float = 60.0,
                 clock=time.monotonic) -> None:
        self.max_calls, self.target_s, self.min_calls = max_calls, target_s, max(1, min_calls)
        self.cap = max(self.min_calls, min(max_calls, start_calls if start_calls is not None else max_calls))   # starts where a fixed cap would be
        self.window_s, self.min_samples, self.adjust_every_s = window_s, min_samples, adjust_every_s
        self.backoff, self.headroom, self.hold_s, self.clock = backoff, headroom, hold_s, clock
        self._after_cut: tuple[int, float] | None = None                   # (the cap before the last cut, the p90 that caused it)
        self._no_cuts_until = -1.0
        self.reverts = 0
        self._samples: deque[tuple[float, float]] = deque(maxlen=2000)    # (when, seconds)
        self._since = clock()                                               # only replies after the last cut count
        self._last_adjust = clock()
        self._lock = threading.Lock()
        self.cuts = self.raises = 0

    def record(self, seconds: float) -> None:
        """One reply's time to the first sound, in seconds."""
        if seconds > 0:
            with self._lock:
                self._samples.append((self.clock(), seconds))

    def p90(self) -> tuple[float | None, int]:
        """The 90th percentile of the recent replies (seconds), and how many there were."""
        with self._lock:
            now = self.clock()
            recent = sorted(s for t, s in self._samples if t >= max(self._since, now - self.window_s))
        if not recent:
            return None, 0
        return recent[min(len(recent) - 1, int(0.9 * len(recent)))], len(recent)

    def adjust(self, active: int) -> int:
        """Move the cap if it is time to (cheap to call often). Returns the cap."""
        now = self.clock()
        if now - self._last_adjust < self.adjust_every_s:
            return self.cap
        self._last_adjust = now
        p90, n = self.p90()
        if p90 is None or n < self.min_samples:
            return self.cap
        if self._after_cut is not None:                                     # the first verdict on the last cut
            before_cap, before_p90 = self._after_cut
            self._after_cut = None
            if p90 > self.target_s and p90 > before_p90 * 0.95:             # not clearly faster: load is not the cause
                self.cap, self.reverts = max(self.cap, before_cap), self.reverts + 1
                self._no_cuts_until = now + self.hold_s
                self._since = now
                return self.cap
        if p90 > self.target_s:
            if now < self._no_cuts_until:
                return self.cap
            new = max(self.min_calls, int(self.cap * self.backoff))
            if new < self.cap:
                self._after_cut = (self.cap, p90)
                self.cap, self.cuts = new, self.cuts + 1
            self._since = now                                           # judge the new cap on fresh replies only
        elif p90 < self.target_s * self.headroom and active >= self.cap - 1 and self.cap < self.max_calls:
            self.cap, self.raises = self.cap + 1, self.raises + 1
        return self.cap

    def accepts(self, active: int) -> bool:
        """May a new call start on this worker now?"""
        return active < self.adjust(active)

    def snapshot(self) -> dict:
        p90, n = self.p90()
        return {"cap": self.cap, "max_calls": self.max_calls, "target_ms": round(self.target_s * 1000), "p90_ms": round(p90 * 1000) if p90 else None,
                "recent_replies": n, "cuts": self.cuts, "raises": self.raises, "reverts": self.reverts,
                "slow_regardless": self.clock() < self._no_cuts_until}
