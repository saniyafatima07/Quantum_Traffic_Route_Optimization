"""Classical Genetic Algorithm, the direct control for QGA.

Same encoding and budget as :class:`QGA`, with the quantum operators replaced by
their standard counterparts: blend crossover and polynomial mutation, selection
by rank, and elitism.  The two implementations differ only in the variation
operators, which is the comparison the report needs.
"""

from __future__ import annotations

import time

import numpy as np

from ...problems.base import Problem
from ..base import Optimizer, OptimizerResult

DEFAULT_POPULATION = 40


class GA(Optimizer):
    """Real-coded GA with blend crossover and polynomial mutation."""

    name = "GA"
    family = "classical"

    def __init__(
        self,
        problem: Problem,
        *,
        budget: int = 4000,
        seed: int = 0,
        population_size: int = DEFAULT_POPULATION,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.15,
        mutation_scale: float = 0.25,
        elite: int = 2,
        sample_every: int = 25,
    ) -> None:
        super().__init__(problem, budget=budget, seed=seed, sample_every=sample_every)
        self.population_size = max(int(population_size), 4)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.mutation_scale = float(mutation_scale)
        self.elite = max(int(elite), 1)

    def _blend_crossover(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """BLX-alpha: sample between and slightly beyond the two parents."""
        alpha = 0.5
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        span = hi - lo
        low = lo - alpha * span
        high = hi + alpha * span
        first = self.rng.random(a.shape) * (high - low) + low
        second = self.rng.random(b.shape) * (high - low) + low
        return np.clip(first, 0.0, 1.0), np.clip(second, 0.0, 1.0)

    def _polynomial_mutation(self, keys: np.ndarray) -> np.ndarray:
        mutated = keys.copy()
        mask = self.rng.random(keys.shape) < self.mutation_rate
        if not mask.any():
            return mutated
        delta = self.rng.normal(0.0, self.mutation_scale, size=int(mask.sum()))
        mutated[mask] = np.clip(mutated[mask] + delta, 0.0, 1.0)
        return mutated

    def run(self) -> OptimizerResult:
        self._started = time.perf_counter()
        keys = np.empty((self.population_size, self.problem.size), dtype=np.float64)
        fitness = np.empty(self.population_size, dtype=np.float64)
        for i in range(self.population_size):
            keys[i] = self.problem.random_keys(self.rng)
            evaluation = self.problem.evaluate_keys(keys[i])
            fitness[i] = evaluation.objective
            self._spend(evaluation, keys[i])

        generation = 0
        while not self.exhausted:
            order = np.argsort(fitness)
            elites = keys[order[: self.elite]].copy()
            elite_fitness = fitness[order[: self.elite]].copy()

            offspring = [e.copy() for e in elites]
            while len(offspring) < self.population_size and not self.exhausted:
                pa, pb = self.rng.choice(self.population_size, size=2, replace=False)
                if self.rng.random() < self.crossover_rate:
                    first, second = self._blend_crossover(keys[pa], keys[pb])
                else:
                    first, second = keys[pa].copy(), keys[pb].copy()
                offspring.append(self._polynomial_mutation(first))
                if len(offspring) < self.population_size:
                    offspring.append(self._polynomial_mutation(second))

            for i, child in enumerate(offspring):
                if self.exhausted:
                    break
                evaluation = self.problem.evaluate_keys(child)
                self._spend(evaluation, child)
                keys[i] = child
                fitness[i] = evaluation.objective

            count = min(self.elite, self.population_size)
            keys[:count] = elites
            fitness[:count] = elite_fitness
            generation += 1

        return self._result(
            {"population": self.population_size, "generations": generation}
        )
