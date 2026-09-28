"""Quantum Particle Swarm Optimization (QPSO).

Following Sun, Zhang and Feng, a particle is not a point in the decision space
but a *quantum probability function*.  One iteration is:

1. **measure** — collapse the particle's Q-bit amplitudes to the real axis with
   ``x = sin^2(theta)``;
2. **mean field** — the swarm's attractor is the global best directly (exploit)
   or a convex blend of the particle's own best and the global best (explore);
3. **collapse** — the amplitude moves to the attractor with probability
   ``w = exp(-|t| |x - mean(pbest)|)``, and otherwise takes a random half-turn
   of phase, which leaves the measured position unchanged for that instant but
   re-spreads the distribution.

Step 3 is what separates this from a classical swarm: a particle that fails to
collapse keeps a *distribution* rather than a value, so diversity is a property
of the state and not just of the spread of positions.  A ``restart`` fraction
re-seeds the worst particles when the swarm stagnates, which is the standard
remedy for the premature convergence a mean-field attractor invites.

Positions are real-valued priority vectors; the problem turns them into routes,
so this drops straight into the same interface as the classical baselines.
"""

from __future__ import annotations

import time

import numpy as np

from ...problems.base import Problem
from ..base import Optimizer, OptimizerResult
from .core import (
    amplitude_from_position,
    bits_of,
    diversity,
    half_turn_phase,
    mean_field,
    measure_position,
    quantum_fluctuation,
)

DEFAULT_SWARM = 24


class QPSO(Optimizer):
    """Quantum-inspired particle swarm over a problem's priority encoding."""

    name = "QPSO"
    family = "quantum"

    def __init__(
        self,
        problem: Problem,
        *,
        budget: int = 4000,
        seed: int = 0,
        swarm_size: int = DEFAULT_SWARM,
        alpha: float = 0.3,
        spread: float = 0.5,
        restart: float = 0.1,
        stagnation_limit: int = 12,
        sample_every: int = 25,
    ) -> None:
        super().__init__(problem, budget=budget, seed=seed, sample_every=sample_every)
        self.swarm_size = max(int(swarm_size), 3)
        self.alpha = float(alpha)
        self.spread = float(spread)
        self.restart_fraction = float(restart)
        self.stagnation_limit = int(stagnation_limit)

    def _init_swarm(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dim = self.problem.size
        keys = np.empty((self.swarm_size, dim), dtype=np.float64)
        theta = np.empty((self.swarm_size, dim), dtype=np.float64)
        fitness = np.empty(self.swarm_size, dtype=np.float64)
        for i in range(self.swarm_size):
            keys[i] = self.problem.random_keys(self.rng)
            theta[i] = amplitude_from_position(keys[i])
            evaluation = self.problem.evaluate_keys(keys[i])
            fitness[i] = evaluation.objective
            self._spend(evaluation, keys[i])
        return keys, theta, fitness

    def run(self) -> OptimizerResult:
        self._started = time.perf_counter()
        keys, theta, fitness = self._init_swarm()

        pbest = keys.copy()
        pbest_fitness = fitness.copy()
        gbest_index = int(np.argmin(pbest_fitness))
        gbest = pbest[gbest_index].copy()
        gbest_fitness = pbest_fitness[gbest_index]

        iteration = 0
        since_improvement = 0
        restarts = 0
        spreads: list[float] = []

        while not self.exhausted:
            mean_pbest = pbest.mean(axis=0)
            for i in range(self.swarm_size):
                if self.exhausted:
                    break
                x = measure_position(theta[i])
                attractor = mean_field(pbest[i], gbest, self.alpha, self.rng)
                w = quantum_fluctuation(x, mean_pbest, iteration, self.spread)
                fallback = half_turn_phase(theta[i], self.rng)
                theta[i] = np.where(self.rng.random(self.problem.size) < w, attractor, fallback)

                x = measure_position(theta[i])
                evaluation = self.problem.evaluate_keys(x)
                self._spend(evaluation, x)
                if evaluation.objective < pbest_fitness[i]:
                    pbest[i] = x
                    pbest_fitness[i] = evaluation.objective
                if evaluation.objective < gbest_fitness:
                    gbest = x.copy()
                    gbest_fitness = evaluation.objective
                    since_improvement = 0
                else:
                    since_improvement += 1

            spreads.append(diversity(bits_of(theta)))
            iteration += 1

            if since_improvement >= self.stagnation_limit and self.restart_fraction > 0:
                restarts += 1
                k = max(int(self.restart_fraction * self.swarm_size), 1)
                for j in np.argsort(-pbest_fitness)[:k]:
                    if self.exhausted:
                        break
                    fresh = self.problem.random_keys(self.rng)
                    theta[j] = amplitude_from_position(fresh)
                    evaluation = self.problem.evaluate_keys(fresh)
                    self._spend(evaluation, fresh)
                    pbest[j] = fresh
                    pbest_fitness[j] = evaluation.objective
                since_improvement = 0

        return self._result(
            {
                "swarm": self.swarm_size,
                "iterations": iteration,
                "restarts": restarts,
                "mean_diversity": round(float(np.mean(spreads)), 4) if spreads else 0.0,
            }
        )


class QPSOWithLocalSearch(QPSO):
    """QPSO hybridised with 2-opt/Or-opt, the usual practical recipe.

    Reported as its own algorithm because the hybrid is what a deployed system
    would actually run; the report keeps both rows so the quantum contribution
    and the local-search contribution stay distinguishable.
    """

    name = "QPSO+LS"
    family = "quantum-hybrid"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from ..classical.local_search import LocalSearchRefiner

        self.refiner = LocalSearchRefiner(self.problem, max_passes=2)

    def run(self) -> OptimizerResult:
        result = super().run()
        if self._best_keys is None:
            return result
        refined = self.refiner.refine(self.problem.decode(self._best_keys))
        evaluation = self.problem.evaluate(refined)
        if evaluation.objective < result.evaluation.objective:
            # Keep the refined routes themselves.  Re-encoding them into the key
            # vector and decoding again would run the split a second time and can
            # hand back a different solution than the one just improved.
            self._best_solution = refined
            self._best_eval = evaluation
            self.convergence.record(self.n_evals, evaluation.objective, self._tick())
        return self._result({**result.meta, "local_search": "2opt-oropt"})
