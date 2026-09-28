"""The optimiser contract and the convergence bookkeeping shared by all of them.

Every algorithm gets the same three things: a problem, a budget of objective
evaluations, and a seeded generator.  In return it reports the best evaluation
it found and, crucially, the *trace* of that search so the report can plot
convergence rather than just a final number.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..problems.base import Evaluation, Problem, Solution


@dataclass(slots=True)
class Convergence:
    """Best-so-far history, sampled at evaluation granularity.

    ``costs[k]`` is the best objective found after the ``k``-th *sampled*
    checkpoint and ``evals[k]`` is how many evaluations had been spent then.
    Sampling on the evaluation axis rather than the iteration axis is what makes
    curves from a swarm, a genetic algorithm and a local search comparable.
    """

    evals: list[int] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    times: list[float] = field(default_factory=list)

    def record(self, n_evals: int, cost: float, elapsed: float) -> None:
        self.evals.append(int(n_evals))
        self.costs.append(float(cost))
        self.times.append(float(elapsed))

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "evals": np.asarray(self.evals, dtype=np.int64),
            "costs": np.asarray(self.costs, dtype=np.float64),
            "times": np.asarray(self.times, dtype=np.float64),
        }

    def iterations_to(self, target: float) -> int | None:
        """Evaluations needed to first reach ``target``; ``None`` if never."""
        for e, c in zip(self.evals, self.costs, strict=True):
            if c <= target:
                return int(e)
        return None

    def final(self) -> float:
        return float(self.costs[-1]) if self.costs else float("inf")


@dataclass(slots=True)
class OptimizerResult:
    """Everything a run produced."""

    algorithm: str
    problem: str
    seed: int
    solution: Solution
    evaluation: Evaluation
    convergence: Convergence
    n_evals: int
    wall_time: float
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "problem": self.problem,
            "seed": self.seed,
            "objective": self.evaluation.objective,
            "time_s": round(self.evaluation.time, 3),
            "distance_m": round(self.evaluation.distance, 2),
            "externality_s": round(self.evaluation.externality, 3),
            "vehicles": self.evaluation.vehicles,
            "penalty": round(self.evaluation.penalty, 3),
            "feasible": self.evaluation.feasible,
            "unserved": self.evaluation.unserved,
            "n_evals": self.n_evals,
            "wall_time_s": round(self.wall_time, 4),
            **self.meta,
        }


class Optimizer(abc.ABC):
    """Base class for every search algorithm in the platform.

    Subclasses implement :meth:`run`; the budget accounting, timing and
    convergence sampling are handled here so the comparison is honest.
    """

    name: str = "abstract"
    family: str = "abstract"

    def __init__(
        self,
        problem: Problem,
        *,
        budget: int = 4000,
        seed: int = 0,
        sample_every: int = 25,
    ) -> None:
        self.problem = problem
        self.budget = int(budget)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.sample_every = max(int(sample_every), 1)
        self.n_evals = 0
        self.convergence = Convergence()
        self._started = 0.0
        self._best_eval: Evaluation | None = None
        self._best_keys: np.ndarray | None = None
        self._best_solution: Solution | None = None

    # -- budget and bookkeeping ---------------------------------------
    def _tick(self) -> float:
        return time.perf_counter() - self._started

    def _spend(
        self,
        evaluation: Evaluation,
        keys: np.ndarray | None = None,
        solution: Solution | None = None,
    ) -> Evaluation:
        """Count one evaluation, keep the incumbent, and sample convergence."""
        self.n_evals += 1
        if self._best_eval is None or evaluation.objective < self._best_eval.objective:
            self._best_eval = evaluation
            if keys is not None:
                self._best_keys = np.array(keys, copy=True)
            if solution is not None:
                self._best_solution = solution
        if self.n_evals % self.sample_every == 0 or self.n_evals == self.budget:
            assert self._best_eval is not None
            self.convergence.record(self.n_evals, self._best_eval.objective, self._tick())
        return evaluation

    @property
    def exhausted(self) -> bool:
        return self.n_evals >= self.budget

    # -- the contract -------------------------------------------------
    @abc.abstractmethod
    def run(self) -> OptimizerResult:
        """Search until the budget is spent, then report the incumbent."""

    def _result(self, meta: dict | None = None) -> OptimizerResult:
        assert self._best_eval is not None
        solution = self._best_solution
        if solution is None:
            solution = self.problem.decode(self._best_keys) if self._best_keys is not None else ()
        # The reported objective has to describe the reported routes.  Decoding a
        # key vector is not guaranteed to rebuild the solution that produced the
        # incumbent evaluation — the savings split and a local search can both
        # come back out of the round trip altered — so re-score whatever is
        # actually being returned instead of trusting a number cached earlier.
        # This is bookkeeping, not search, so it is deliberately not charged to
        # the evaluation budget.
        evaluation = self.problem.evaluate(solution)
        return OptimizerResult(
            algorithm=self.name,
            problem=self.problem.name,
            seed=self.seed,
            solution=solution,
            evaluation=evaluation,
            convergence=self.convergence,
            n_evals=self.n_evals,
            wall_time=self._tick(),
            meta=dict(meta or {}),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.__class__.__name__}(problem={self.problem.name}, budget={self.budget})"


def run_optimizer(
    factory,
    problem: Problem,
    *,
    budget: int,
    seeds: Sequence[int],
    sample_every: int = 25,
) -> list[OptimizerResult]:
    """Run one algorithm over several seeds and collect the results."""
    results: list[OptimizerResult] = []
    for seed in seeds:
        optimizer = factory(problem, budget=budget, seed=seed, sample_every=sample_every)
        results.append(optimizer.run())
    return results
