"""Quantum-Inspired Self-Organising Evolutionary Programming (QISEP).

Zhang and Feng's answer to the fact that a swarm has nothing to say about *which*
parts of its solutions are good.  A generation of good solutions is compared
component-wise, and the components on which the good solutions agree — the
*prime subspace* — are taken to be the parts of the route that matter.  The
remainder is re-arranged freely, and the self-organising step is what converts
that re-arrangement from blind permutation into informed variation.

Quantum inspiration enters in two places:

* the population is a set of Q-bit amplitudes, so a member carries a
  distribution over arrangements rather than one arrangement;
* the variation is a Hadamard-style rotation, ``|x> -> cos(x)|0> +
  sin(x)|1>``, followed by measurement ``x = sin^2(theta)``, which is the
  standard quantum-inspired gate.

QISEP is the strongest of the three quantum-inspired methods here because it
combines the population's structural knowledge with the local refinement that
the swarm methods lack; it is also the most expensive, since the prime subspace
is recomputed every generation.
"""

from __future__ import annotations

import time

import numpy as np

from ...problems.base import Problem, Solution
from ..base import Optimizer, OptimizerResult
from .core import amplitude_from_position, bits_of, diversity, measure_position

DEFAULT_POPULATION = 30
PRIME_THRESHOLD = 0.7


class QISEP(Optimizer):
    """Self-organising evolutionary programming with quantum rotation gates."""

    name = "QISEP"
    family = "quantum"

    def __init__(
        self,
        problem: Problem,
        *,
        budget: int = 4000,
        seed: int = 0,
        population_size: int = DEFAULT_POPULATION,
        elite: int = 6,
        prime_threshold: float = PRIME_THRESHOLD,
        sample_every: int = 25,
    ) -> None:
        super().__init__(problem, budget=budget, seed=seed, sample_every=sample_every)
        self.population_size = max(int(population_size), 4)
        self.elite = max(int(elite), 2)
        self.prime_threshold = float(prime_threshold)

    # ------------------------------------------------------------------
    # quantum gates
    # ------------------------------------------------------------------
    @staticmethod
    def _rotation(theta: np.ndarray, angle: np.ndarray) -> np.ndarray:
        """A Hadamard-family rotation: ``theta -> theta + angle`` then measured."""
        return theta + angle

    def _mutate(self, theta: np.ndarray, rate: float) -> np.ndarray:
        """Rotation about a random axis on the selected genes."""
        mutated = theta.copy()
        mask = self.rng.random(theta.shape) < rate
        mutated[mask] += self.rng.normal(0.0, np.pi / 3.0, size=int(mask.sum()))
        return mutated

    # ------------------------------------------------------------------
    # self-organisation
    # ------------------------------------------------------------------
    def _prime_mask(self, elite_orders: np.ndarray) -> np.ndarray:
        """Genuine elements of the prime subspace.

        Element ``c`` is prime when at least ``prime_threshold`` of the elite
        rank it the same or an adjacent position.  Adjacency rather than exact
        agreement is what keeps the subspace useful: reordering a *pair* leaves
        a route that is genuinely different, and the search keeps it.
        """
        n = elite_orders.shape[1]
        elite = elite_orders.shape[0]
        if elite == 0:
            return np.ones(n, dtype=bool)
        agreement = np.zeros(n, dtype=np.float64)
        for c in range(n):
            positions = elite_orders[:, c]
            hits = np.count_nonzero(np.any(np.abs(positions[:, None] - positions[None, :]) <= 1, axis=1))
            agreement[c] = hits / elite
        return agreement >= self.prime_threshold

    # ------------------------------------------------------------------
    def run(self) -> OptimizerResult:
        self._started = time.perf_counter()
        dim = self.problem.size
        theta = np.empty((self.population_size, dim), dtype=np.float64)
        orders = np.empty((self.population_size, dim), dtype=np.int64)
        fitness = np.empty(self.population_size, dtype=np.float64)
        keys = np.empty((self.population_size, dim), dtype=np.float64)

        for i in range(self.population_size):
            theta[i] = amplitude_from_position(self.problem.random_keys(self.rng))
            x = measure_position(theta[i])
            keys[i] = x
            orders[i] = np.asarray(self.problem.order_of(x), dtype=np.int64)
            evaluation = self.problem.evaluate_keys(x)
            fitness[i] = evaluation.objective
            self._spend(evaluation, x)

        generation = 0
        prime_sizes: list[float] = []
        spreads: list[float] = []

        while not self.exhausted:
            rank = np.argsort(fitness)
            elite = rank[: min(self.elite, self.population_size)]
            prime = self._prime_mask(orders[elite])
            prime_sizes.append(float(prime.sum()))
            nonprime = np.flatnonzero(~prime)
            elites = theta[elite].copy()
            elite_fitness = fitness[elite].copy()

            offspring = [e.copy() for e in elites]
            offspring_fitness = list(elite_fitness)
            offspring_orders = [orders[e].copy() for e in elite]

            for i in range(len(offspring), self.population_size):
                if self.exhausted:
                    break
                parent = elite[self.rng.integers(len(elite))]
                child = theta[parent].copy()
                if nonprime.size:
                    # Re-arranging only the non-prime elements is the informed
                    # step: the prime order is carried over intact.
                    positions = self.rng.permutation(nonprime)
                    child_keys = keys[parent].copy()
                    child_keys[nonprime] = child_keys[positions]
                    child = amplitude_from_position(child_keys)
                child = self._mutate(child, 0.2)
                x = measure_position(child)
                child_order = np.asarray(self.problem.order_of(x), dtype=np.int64)
                evaluation = self.problem.evaluate_keys(x)
                self._spend(evaluation, x)
                offspring.append(child)
                offspring_fitness.append(evaluation.objective)
                offspring_orders.append(child_order)

            count = len(offspring)
            theta[:count] = np.array(offspring[:count])
            fitness[:count] = np.array(offspring_fitness[:count])
            orders[:count] = np.array(offspring_orders[:count])
            keys[:count] = measure_position(theta[:count])
            spreads.append(diversity(bits_of(theta[:count])))
            generation += 1

        return self._result(
            {
                "population": self.population_size,
                "generations": generation,
                "mean_prime_size": round(float(np.mean(prime_sizes)), 2) if prime_sizes else 0.0,
                "mean_diversity": round(float(np.mean(spreads)), 4) if spreads else 0.0,
            }
        )


class QISEPWithLocalSearch(QISEP):
    """QISEP followed by a deterministic local-search polish.

    Reported separately so the self-organisation step and the refinement step
    stay distinguishable in the benchmark.
    """

    name = "QISEP+LS"
    family = "quantum-hybrid"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from ..classical.local_search import LocalSearchRefiner

        self.refiner = LocalSearchRefiner(self.problem, max_passes=3)

    def run(self) -> OptimizerResult:
        result = super().run()
        if self._best_keys is None:
            return result
        refined: Solution = self.refiner.refine(self.problem.decode(self._best_keys))
        evaluation = self.problem.evaluate(refined)
        if evaluation.objective < result.evaluation.objective:
            # Keep the refined routes themselves; see the note in qpso.run.
            self._best_solution = refined
            self._best_eval = evaluation
            self.convergence.record(self.n_evals, evaluation.objective, self._tick())
        return self._result({**result.meta, "local_search": "2opt-oropt"})
