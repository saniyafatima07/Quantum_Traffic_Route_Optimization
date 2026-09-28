"""Classical Particle Swarm Optimization, the direct control for QPSO.

Same encoding, same budget, same evaluation function as :class:`QPSO`; the only
differences are the position update (velocity plus inertia instead of a
quantum collapse) and the absence of a distribution, since a classical particle
*is* a point.  Holding everything else fixed is what makes the pair a fair
comparison of the search mechanism rather than of the tuning.
"""

from __future__ import annotations

import time

import numpy as np

from ...problems.base import Problem
from ..base import Optimizer, OptimizerResult

DEFAULT_SWARM = 24


class PSO(Optimizer):
    """Textbook PSO with inertia weight and constriction-free velocity clamping."""

    name = "PSO"
    family = "classical"

    def __init__(
        self,
        problem: Problem,
        *,
        budget: int = 4000,
        seed: int = 0,
        swarm_size: int = DEFAULT_SWARM,
        inertia: float = 0.72,
        cognitive: float = 1.49,
        social: float = 1.49,
        v_clamp: float = 0.5,
        sample_every: int = 25,
    ) -> None:
        super().__init__(problem, budget=budget, seed=seed, sample_every=sample_every)
        self.swarm_size = max(int(swarm_size), 3)
        self.inertia = float(inertia)
        self.cognitive = float(cognitive)
        self.social = float(social)
        self.v_clamp = float(v_clamp)

    def run(self) -> OptimizerResult:
        self._started = time.perf_counter()
        dim = self.problem.size
        keys = np.empty((self.swarm_size, dim), dtype=np.float64)
        velocity = np.empty((self.swarm_size, dim), dtype=np.float64)
        fitness = np.empty(self.swarm_size, dtype=np.float64)

        for i in range(self.swarm_size):
            keys[i] = self.problem.random_keys(self.rng)
            velocity[i] = self.rng.uniform(-self.v_clamp, self.v_clamp, size=dim)
            evaluation = self.problem.evaluate_keys(keys[i])
            fitness[i] = evaluation.objective
            self._spend(evaluation, keys[i])

        pbest = keys.copy()
        pbest_fitness = fitness.copy()
        gbest_index = int(np.argmin(pbest_fitness))
        gbest = pbest[gbest_index].copy()
        gbest_fitness = pbest_fitness[gbest_index]

        iteration = 0
        while not self.exhausted:
            for i in range(self.swarm_size):
                if self.exhausted:
                    break
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                velocity[i] = (
                    self.inertia * velocity[i]
                    + self.cognitive * r1 * (pbest[i] - keys[i])
                    + self.social * r2 * (gbest - keys[i])
                )
                np.clip(velocity[i], -self.v_clamp, self.v_clamp, out=velocity[i])
                keys[i] = np.clip(keys[i] + velocity[i], 0.0, 1.0)

                evaluation = self.problem.evaluate_keys(keys[i])
                self._spend(evaluation, keys[i])
                if evaluation.objective < pbest_fitness[i]:
                    pbest[i] = keys[i]
                    pbest_fitness[i] = evaluation.objective
                if evaluation.objective < gbest_fitness:
                    gbest = keys[i].copy()
                    gbest_fitness = evaluation.objective
            iteration += 1

        return self._result({"swarm": self.swarm_size, "iterations": iteration})
