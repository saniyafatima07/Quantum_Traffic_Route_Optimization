"""Simulated annealing, the control for the local-search hybrids.

Each step perturbs the priority vector by a random-walk kick whose size is the
current temperature, accepts an improvement always and a worsening move with
probability ``exp(-delta / T)``, and cools geometrically.  It is the cheapest
reasonable baseline that has no population at all, so it isolates how much of
QPSO's performance comes from the swarm rather than from the neighbourhood.
"""

from __future__ import annotations

import time

import numpy as np

from ...problems.base import Problem
from ..base import Optimizer, OptimizerResult


class SimulatedAnnealing(Optimizer):
    """Geometric-cooling simulated annealing over the priority encoding."""

    name = "SA"
    family = "classical"

    def __init__(
        self,
        problem: Problem,
        *,
        budget: int = 4000,
        seed: int = 0,
        initial_temperature: float = 0.35,
        cooling_rate: float = 0.995,
        step_size: float = 0.25,
        sample_every: int = 25,
    ) -> None:
        super().__init__(problem, budget=budget, seed=seed, sample_every=sample_every)
        self.initial_temperature = float(initial_temperature)
        self.cooling_rate = float(cooling_rate)
        self.step_size = float(step_size)

    def run(self) -> OptimizerResult:
        self._started = time.perf_counter()
        current = self.problem.random_keys(self.rng)
        current_evaluation = self.problem.evaluate_keys(current)
        self._spend(current_evaluation, current)

        best_keys = current.copy()
        best_evaluation = current_evaluation
        temperature = self.initial_temperature
        steps = 0
        accepted = 0

        while not self.exhausted:
            trial = np.clip(
                current + self.rng.normal(0.0, self.step_size, size=self.problem.size),
                0.0,
                1.0,
            )
            evaluation = self.problem.evaluate_keys(trial)
            self._spend(evaluation, trial)
            steps += 1
            delta = evaluation.objective - current_evaluation.objective
            if delta < 0 or self.rng.random() < np.exp(-delta / max(temperature, 1e-9)):
                current, current_evaluation = trial, evaluation
                accepted += 1
                if evaluation.objective < best_evaluation.objective:
                    best_keys = trial.copy()
                    best_evaluation = evaluation
                    self._best_keys = best_keys.copy()
                    self._best_eval = best_evaluation
            temperature *= self.cooling_rate

        return self._result(
            {
                "steps": steps,
                "acceptance": round(accepted / max(steps, 1), 4),
                "final_temperature": round(temperature, 6),
            }
        )
