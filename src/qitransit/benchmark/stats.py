"""Statistical comparison across methods.

Metaheuristic comparisons are routinely reported as a mean and a standard
deviation, which says very little: on a hard instance a method's spread across
seeds can exceed the gap between methods, and a mean alone cannot tell a
genuinely better method from a lucky one.  These are the two tests that can.

* **Friedman** over the whole grid, to ask whether *any* method differs.
* **Wilcoxon signed-rank** pairwise against a chosen reference, to ask *which*
  differ, with a Holm correction so that testing many pairs does not manufacture
  significance.

Both are non-parametric and work on ranks, which is what makes them the right
default: objective values across instances are heavy-tailed and on wildly
different scales, and a t-test's normality assumption is the first thing to go.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class TestResult:
    """The outcome of one non-parametric test."""

    name: str
    statistic: float
    p_value: float
    n: int
    detail: dict

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "statistic": round(float(self.statistic), 6),
            "p_value": float(f"{self.p_value:.3e}"),
            "n": self.n,
            "significant": self.significant,
            **self.detail,
        }


def _grid(records, key) -> tuple[list[str], np.ndarray]:
    """A ``[methods x (instance, seed)]`` matrix of objectives.

    Instances and seeds are paired on both axes so that a pairwise test compares
    like with like.  Cells where one method has no run are dropped rather than
    imputed: an imputed value would be a number nobody measured.
    """
    methods = sorted({r.method for r in records})
    keys = sorted({(r.problem_family, r.instance, r.seed) for r in records})
    index = {key: i for i, key in enumerate(keys)}
    table = np.full((len(methods), len(keys)), np.nan, dtype=np.float64)
    for record in records:
        row = methods.index(record.method)
        table[row, index[(record.problem_family, record.instance, record.seed)]] = record.objective
    keep = ~np.isnan(table).any(axis=0)
    return methods, table[:, keep]


def friedman(records, key: str = "objective") -> TestResult:
    """Is any method better than another, across the whole grid?"""
    methods, table = _grid(records, key)
    ranks = stats.rankdata(table, axis=1)
    statistic, p_value = stats.friedmanchisquare(*ranks)
    return TestResult(
        name="friedman",
        statistic=float(statistic),
        p_value=float(p_value),
        n=table.shape[1],
        detail={
            "methods": methods,
            "mean_ranks": [round(float(r), 4) for r in ranks.mean(axis=1)],
            "metric": key,
        },
    )


def wilcoxon_vs_reference(
    records, reference: str, key: str = "objective", alpha: float = 0.05
) -> list[TestResult]:
    """Every other method against ``reference``, with a Holm correction.

    Signed-rank rather than signed-wilcoxon: it assumes only symmetry of the
    paired differences, which is a much smaller ask than normality for a
    difference of two optimisers' objectives.
    """
    methods, table = _grid(records, key)
    if reference not in methods:
        raise KeyError(f"reference {reference!r} not among {methods}")
    base = table[methods.index(reference)]
    results: list[TestResult] = []
    raw: list[float] = []
    for method in methods:
        if method == reference:
            continue
        other = table[methods.index(method)]
        delta = other - base
        if np.allclose(delta, 0.0):
            raw.append(1.0)
            results.append(
                TestResult(
                    name=f"wilcoxon:{method}",
                    statistic=0.0,
                    p_value=1.0,
                    n=int(delta.size),
                    detail={"reference": reference, "identical": True},
                )
            )
            continue
        statistic, p_value = stats.wilcoxon(other, base, alternative="two-sided")
        raw.append(float(p_value))
        results.append(
            TestResult(
                name=f"wilcoxon:{method}",
                statistic=float(statistic),
                p_value=float(p_value),
                n=int(delta.size),
                detail={
                    "reference": reference,
                    "median_difference": round(float(np.median(delta)), 4),
                    "mean_difference": round(float(delta.mean()), 4),
                },
            )
        )
    corrected = holm(raw, alpha)
    return [
        TestResult(r.name, r.statistic, p, r.n, {**r.detail, "p_holm": p})
        for r, p in zip(results, corrected, strict=True)
    ]


def holm(p_values: Sequence[float], alpha: float = 0.05) -> list[float]:
    """Holm–Bonferroni adjusted p-values, order preserved."""
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    m = values.size
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def effect_size(table: np.ndarray) -> np.ndarray:
    """Cliff's delta per method against the per-column best.

    Reported because a p-value says whether a difference exists and says nothing
    about whether it is worth having; the delta says which direction and how far
    from "no difference" the ranking sits.
    """
    best = table.min(axis=0)
    out = np.empty(table.shape[0], dtype=np.float64)
    for i in range(table.shape[0]):
        greater = np.sum(table[i] > best)
        lesser = np.sum(table[i] < best)
        total = greater + lesser
        out[i] = 0.0 if total == 0 else (lesser - greater) / total
    return out


def summarise_tests(records, reference: str, key: str = "objective") -> dict:
    methods, table = _grid(records, key)
    delta = effect_size(table)
    return {
        "friedman": friedman(records, key).as_dict(),
        "wilcoxon": [t.as_dict() for t in wilcoxon_vs_reference(records, reference, key)],
        "reference": reference,
        "cliffs_delta": {
            method: round(float(d), 4) for method, d in zip(methods, delta, strict=True)
        },
    }
