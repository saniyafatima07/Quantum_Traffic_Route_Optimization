"""Quantum Genetic Algorithm (QGA).

Each gene is a Q-bit whose amplitude is the state the population stores, so a
chromosome is a distribution over whole solutions rather than one solution.  The
operators are the standard quantum-inspired ones:

* **quantum crossover** — the child amplitude is a linear combination
  ``alpha * |parent_a> + (1 - alpha) * |parent_b>``, and the gene is measured
  against that combination's own amplitude, so the child is a genuinely new
  point rather than a copy of either parent;
* **quantum mutation** — a rotation of the selected gene about a random axis,
  which re-expands the local distribution;
* **measurement** — ``x = sin^2(theta)`` collapses the chromosome to a
  classical priority vector.

Crossover exploring the *space between* two good solutions, rather than picking
one of them, is the behaviour that distinguishes this population from a
classical one.
"""

from __future__ import annotations

import time

import numpy as np

from ...problems.base import Problem
from ..base import Optimizer, OptimizerResult
from .core import amplitude_from_position, bits_of, diversity, measure_position

DEFAULT_POPULATION = 40


class QGA(Optimizer):
    """Quantum-inspired genetic algorithm over the priority encoding."""

    name = "QGA"
    family = "quantum"

    def __init__(
        self,
        problem: Problem,
        *,
        budget: int = 4000,
        seed: int = 0,
        population_size: int = DEFAULT_POPULATION,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.15,
        elite: int = 2,
        sample_every: int = 25,
    ) -> None:
        super().__init__(problem, budget=budget, seed=seed, sample_every=sample_every)
        self.population_size = max(int(population_size), 4)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.elite = max(int(elite), 1)

    def _init_population(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dim = self.problem.size
        theta = np.empty((self.population_size, dim), dtype=np.float64)
        fitness = np.empty(self.population_size, dtype=np.float64)
        for i in range(self.population_size):
            theta[i] = amplitude_from_position(self.problem.random_keys(self.rng))
            x = measure_position(theta[i])
            evaluation = self.problem.evaluate_keys(x)
            fitness[i] = evaluation.objective
            self._spend(evaluation, x)
        return theta, fitness

    def _quantum_crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Superpose two parents' amplitudes and measure the interference.

        The child's amplitude is the linear combination; the gene is then drawn
        with probability ``sin^2`` of that amplitude, which is the classical
        shadow of an interference measurement.
        """
        alpha = self.rng.random()
        return alpha * a + (1.0 - alpha) * b

    def _quantum_mutation(self, theta: np.ndarray) -> np.ndarray:
        """Rotate the selected genes, re-expanding their local distribution."""
        mutated = theta.copy()
        mask = self.rng.random(theta.shape) < self.mutation_rate
        if not mask.any():
            return mutated
        angle = self.rng.normal(0.0, np.pi / 4.0, size=theta.shape)
        mutated[mask] += angle[mask]
        return mutated

    def run(self) -> OptimizerResult:
        self._started = time.perf_counter()
        theta, fitness = self._init_population()
        generation = 0
        spreads: list[float] = []

        while not self.exhausted:
            order = np.argsort(fitness)
            elites = theta[order[: self.elite]].copy()
            elite_fitness = fitness[order[: self.elite]].copy()

            offspring = [e.copy() for e in elites]
            while len(offspring) < self.population_size and not self.exhausted:
                pa, pb = self.rng.choice(self.population_size, size=2, replace=False)
                if self.rng.random() < self.crossover_rate:
                    child = self._quantum_crossover(theta[pa], theta[pb])
                else:
                    child = theta[pa].copy()
                offspring.append(self._quantum_mutation(child))

            for i, child_theta in enumerate(offspring):
                if self.exhausted:
                    break
                x = measure_position(child_theta)
                evaluation = self.problem.evaluate_keys(x)
                self._spend(evaluation, x)
                theta[i] = child_theta
                fitness[i] = evaluation.objective

            theta[: self.elite] = elites
            fitness[: self.elite] = elite_fitness
            spreads.append(diversity(bits_of(theta)))
            generation += 1

        return self._result(
            {
                "population": self.population_size,
                "generations": generation,
                "mean_diversity": round(float(np.mean(spreads)), 4) if spreads else 0.0,
            }
        )
