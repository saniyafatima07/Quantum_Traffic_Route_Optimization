"""Clarke-Wright savings exposed as an optimiser, for the benchmark's floor row.

The construction itself lives in :mod:`qitransit.problems.fleet` because it is a
decoding strategy as much as an algorithm: the problem's own ``decode`` uses the
same routine, which is exactly why the baseline is a fair reference for it.
"""

from __future__ import annotations

import time

from ...problems.base import Problem, Solution
from ...problems.fleet import clarke_wright
from ..base import Optimizer, OptimizerResult


class SavingsBaseline(Optimizer):
    """Deterministic savings construction, charged a single evaluation."""

    name = "Savings"
    family = "classical"

    def __init__(
        self,
        problem: Problem,
        *,
        budget: int = 1,
        seed: int = 0,
        sample_every: int = 1,
    ) -> None:
        super().__init__(problem, budget=budget, seed=seed, sample_every=sample_every)

    def run(self) -> OptimizerResult:
        self._started = time.perf_counter()
        solution: Solution = clarke_wright(self.problem)
        evaluation = self.problem.evaluate(solution)
        self._spend(evaluation, self.problem.encode(solution), solution)
        return self._result({"deterministic": True})
