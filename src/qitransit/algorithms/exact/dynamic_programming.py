"""Exact solution of small instances, used to calibrate the metaheuristics.

Held-Karp dynamic programming gives the true optimum of a TSP with ``n``
customers in ``O(n^2 2^n)`` time and ``O(n 2^n)`` memory.  It is included not
because it scales to the Bengaluru network — nothing exact does — but because an
optimum is the only way to say whether a metaheuristic is *good*, as opposed
merely better than its rivals.

The recurrence used here is the backward one:

.. math::

    D[S][j] = \\min_{k \\in S \\setminus \\{j\\}}
              \\Big( D[S \\setminus \\{j\\}][k] + c_{k,j} \\Big),
    \\qquad
    D[\\{j\\}][j] = c_{0j}

with the tour closed by minimising ``D[V][j] + c_{j0}`` over the last customer
``j``.  Removing a set bit always decreases the mask, so processing masks in
increasing numeric order is a valid topological order.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from ...problems.base import Evaluation

DEFAULT_LIMIT = 16


@dataclass(slots=True)
class ExactResult:
    """The optimum, and the evidence for claiming it is one."""

    cost: float
    order: tuple[int, ...]
    evaluations: int
    wall_time: float
    exhaustive: bool = True
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "cost": self.cost,
            "order": list(self.order),
            "evaluations": self.evaluations,
            "wall_time_s": round(self.wall_time, 4),
            "exhaustive": self.exhaustive,
            **self.meta,
        }


def held_karp(cost: np.ndarray, max_customers: int = DEFAULT_LIMIT) -> ExactResult:
    """The optimal tour through customers ``1..n`` of a complete cost matrix.

    ``cost[0]`` is the depot row.  Beyond ``max_customers`` the instance is
    rejected rather than silently truncated, because a truncated DP would return
    a confidently wrong answer.
    """
    n = int(cost.shape[0]) - 1
    if n == 0:
        return ExactResult(0.0, (), 0, 0.0, meta={"customers": 0})
    if n > max_customers:
        raise ValueError(f"Held-Karp is limited to {max_customers} customers, got {n}")

    started = time.perf_counter()
    m = n
    size = 1 << m
    best = np.full((size, n), np.inf, dtype=np.float64)
    parent = np.full((size, n), -1, dtype=np.int32)
    cost_row = np.ascontiguousarray(cost[1:, 1:], dtype=np.float64)
    depot = cost[0, 1:].astype(np.float64)

    members: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * size
    for mask in range(1, size):
        low = mask & -mask
        members[mask] = np.sort(np.append(members[mask ^ low], low.bit_length() - 1))

    evaluations = 0
    for mask in range(1, size):
        for j in members[mask]:
            previous = mask ^ (1 << j)
            if previous == 0:
                best[mask, j] = depot[j]
                evaluations += 1
                continue
            candidates = members[previous]
            totals = best[previous, candidates] + cost_row[candidates, j]
            evaluations += candidates.size
            if not np.isfinite(totals).any():
                continue
            best[mask, j] = totals.min()
            parent[mask, j] = candidates[int(np.argmin(totals))]

    full = size - 1
    last_members = members[full]
    closing = best[full, last_members] + cost[last_members + 1, 0]
    last = int(last_members[int(np.argmin(closing))])
    optimum = float(closing.min())

    order: list[int] = []
    mask = full
    j: int = last
    while j >= 0:
        order.append(j)
        previous_j = int(parent[mask, j])
        mask ^= 1 << j
        j = previous_j
    order.reverse()

    return ExactResult(
        cost=optimum,
        order=tuple(order),
        evaluations=evaluations,
        wall_time=time.perf_counter() - started,
        meta={"customers": n, "states": int(size)},
    )


def brute_force_tsp(cost: np.ndarray) -> ExactResult:
    """Exhaustive enumeration; only tractable for a handful of customers.

    Kept because it shares no logic with :func:`held_karp` and therefore serves
    as an independent check on the DP's value and reconstruction.
    """
    n = int(cost.shape[0]) - 1
    if n > 9:
        raise ValueError(f"brute force is limited to 9 customers, got {n}")
    started = time.perf_counter()
    if n == 0:
        return ExactResult(0.0, (), 0, time.perf_counter() - started, meta={"customers": 0})

    best_cost = np.inf
    best_order: tuple[int, ...] = ()
    evaluations = 0
    for permutation in itertools.permutations(range(n)):
        total = cost[0, permutation[0] + 1]
        for a, b in zip(permutation, permutation[1:], strict=False):
            total += cost[a + 1, b + 1]
        total += cost[permutation[-1] + 1, 0]
        evaluations += 1
        if total < best_cost:
            best_cost = float(total)
            best_order = permutation
    return ExactResult(
        cost=best_cost,
        order=best_order,
        evaluations=evaluations,
        wall_time=time.perf_counter() - started,
        meta={"customers": n},
    )


def as_evaluation(result: ExactResult, problem) -> Evaluation:
    """Score an exact tour through the same evaluator the optimisers use."""
    solution = (tuple(c - 1 for c in result.order),)
    return problem.evaluate(solution)
