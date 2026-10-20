"""Test-drive slot rules: per-branch working hours, a fixed grid, a minimum lead time and a booking horizon."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

from ..config import BranchCfg, Config
from ..models import Car
from ..repos.base import BookingRepository

_WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


class SlotService:
    def __init__(self, cfg: Config, bookings: BookingRepository, clock: Callable[[], datetime]) -> None:
        self.cfg, self.bookings, self.clock = cfg, bookings, clock
        self._branches = {b.id: b for b in cfg.branches}

    def branch(self, branch_id: str) -> BranchCfg:
        return self._branches[branch_id]

    def tz(self, branch_id: str) -> ZoneInfo:
        return ZoneInfo(self.branch(branch_id).timezone)

    def grid(self, branch_id: str, day: date) -> list[datetime]:
        """Every slot start on the grid for that local day (timezone-aware), ignoring bookings and the clock."""
        b, tz = self.branch(branch_id), self.tz(branch_id)
        hours = b.working_hours.get(_WEEKDAYS[day.weekday()])
        if not hours:
            return []                                                       # closed that day
        opens, closes = (datetime.combine(day, time.fromisoformat(h), tzinfo=tz) for h in hours)
        step = timedelta(minutes=self.cfg.test_drive.slot_minutes)
        out, t = [], opens
        while t + step <= closes:
            out.append(t)
            t += step
        return out

    def _earliest(self) -> datetime:
        return self.clock() + timedelta(minutes=self.cfg.test_drive.min_lead_minutes)

    def _latest_day(self, branch_id: str) -> date:
        return (self.clock().astimezone(self.tz(branch_id)) + timedelta(days=self.cfg.test_drive.horizon_days)).date()

    def invalid_reason(self, branch_id: str, start: datetime) -> str | None:
        """None if `start` is a bookable slot start at this branch, otherwise a short reason."""
        if start.tzinfo is None:
            return "naive_datetime"
        local = start.astimezone(self.tz(branch_id))
        if start < self._earliest():
            return "too_soon"
        if local.date() > self._latest_day(branch_id):
            return "too_far_ahead"
        grid = self.grid(branch_id, local.date())
        if not grid:
            return "branch_closed"
        if local not in grid:
            return "outside_hours_or_off_grid"
        return None

    def free_slots(self, car: Car, day: date) -> list[datetime]:
        grid = [s for s in self.grid(car.branch_id, day) if self.invalid_reason(car.branch_id, s) is None]
        if not grid:
            return []
        taken = self.bookings.booked_starts(car.branch_id, grid[0], grid[-1] + timedelta(minutes=1))
        return [s for s in grid if s not in taken]

    def alternatives(self, car: Car, near: datetime, k: int = 3) -> list[datetime]:
        """The next `k` free slots at or after `near` (same day first, then following days within the horizon)."""
        if near.tzinfo is None:                 # a programming error upstream: refuse it, but still offer real slots
            near = self.clock()
        day = near.astimezone(self.tz(car.branch_id)).date()
        out: list[datetime] = []
        for offset in range(self.cfg.test_drive.horizon_days + 1):
            for s in self.free_slots(car, day + timedelta(days=offset)):
                if s >= near:
                    out.append(s)
                    if len(out) == k:
                        return out
        return out
