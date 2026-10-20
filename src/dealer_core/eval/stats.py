"""Paired statistics for comparing two systems on the SAME utterances (pure Python, seeded)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from math import comb


@dataclass(frozen=True)
class PairedResult:
    n: int
    diff_pts: float          # candidate minus baseline, percentage points
    ci_low: float            # 95% bootstrap interval, percentage points
    ci_high: float
    cand_only: int           # McNemar: candidate right, baseline wrong
    base_only: int           # baseline right, candidate wrong
    p_mcnemar: float         # exact two-sided


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_compare(cand: list[bool], base: list[bool], resamples: int = 10_000, seed: int = 7) -> PairedResult:
    assert len(cand) == len(base) and cand, "paired lists must be the same non-zero length"
    n = len(cand)
    d = [int(c) - int(b) for c, b in zip(cand, base)]
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    up, down = sum(1 for x in d if x == 1), sum(1 for x in d if x == -1)
    return PairedResult(n=n, diff_pts=100 * sum(d) / n, ci_low=100 * means[int(0.025 * resamples)],
                        ci_high=100 * means[int(0.975 * resamples)], cand_only=up, base_only=down,
                        p_mcnemar=mcnemar_exact(up, down))


def gate_verdict(r: PairedResult, max_drop_pts: float = 5.0) -> str:
    """The T0 decision rule: a drop is CONFIRMED only if the whole 95% interval lies below -max_drop_pts."""
    if r.ci_high < -max_drop_pts:
        return "DROP CONFIRMED: stop and report before building on the candidate"
    if r.ci_low > -max_drop_pts:
        return "NO LARGE DROP: the candidate is within the threshold"
    return ("INCONCLUSIVE: the interval spans the threshold. Proceed with the candidate for the voice path, keep the "
            "baseline as the configured fallback, and re-run when the labelled set grows")
