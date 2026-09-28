"""The algorithm catalogue and the registry the benchmark schedules from.

Keeping the registry in one place means the CLI, the benchmark harness and the
report all refer to the same names, so a method cannot be quietly added to one
and forgotten in another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

from ..problems.base import Problem
from .base import Convergence, Optimizer, OptimizerResult, run_optimizer
from .classical import GA, PSO, LocalSearchRefiner, SavingsBaseline, SimulatedAnnealing
from .exact import brute_force_tsp, compare_exact_searchers, exact_shortest_path, held_karp
from .quantum import QGA, QISEP, QPSO, QISEPWithLocalSearch, QPSOWithLocalSearch

OptimizerFactory = Callable[..., Optimizer]


@dataclass(frozen=True, slots=True)
class Entry:
    """One schedulable method."""

    key: str
    name: str
    family: str
    factory: OptimizerFactory
    tunable: tuple[str, ...] = ()

    def build(
        self, problem: Problem, *, budget: int, seed: int, sample_every: int = 25, **params
    ) -> Optimizer:
        return self.factory(
            problem, budget=budget, seed=seed, sample_every=sample_every, **params
        )


REGISTRY: dict[str, Entry] = {}


def _register(entry: Entry) -> Entry:
    REGISTRY[entry.key] = entry
    return entry


_register(Entry("qpso", "QPSO", "quantum", QPSO, ("swarm_size", "alpha", "spread", "restart")))
_register(Entry("qpso-ls", "QPSO+LS", "quantum-hybrid", QPSOWithLocalSearch, ("swarm_size",)))
_register(Entry("qga", "QGA", "quantum", QGA, ("population_size", "crossover_rate", "mutation_rate")))
_register(Entry("qisep", "QISEP", "quantum", QISEP, ("population_size", "prime_threshold")))
_register(Entry("qisep-ls", "QISEP+LS", "quantum-hybrid", QISEPWithLocalSearch, ("population_size",)))
_register(Entry("pso", "PSO", "classical", PSO, ("swarm_size", "inertia", "cognitive", "social")))
_register(Entry("ga", "GA", "classical", GA, ("population_size", "crossover_rate", "mutation_rate")))
_register(Entry("sa", "SA", "classical", SimulatedAnnealing, ("initial_temperature", "cooling_rate")))
_register(Entry("savings", "Savings", "classical", SavingsBaseline, ()))


def quantum_keys() -> list[str]:
    return [k for k, e in REGISTRY.items() if e.family.startswith("quantum")]


def classical_keys() -> list[str]:
    return [k for k, e in REGISTRY.items() if e.family == "classical"]


def all_keys() -> list[str]:
    return list(REGISTRY)


def _normalise(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace("+", "p")


def get(key: str) -> Entry:
    wanted = _normalise(key)
    for candidate, entry in REGISTRY.items():
        if _normalise(candidate) == wanted or _normalise(entry.name) == wanted:
            return entry
    raise KeyError(f"unknown algorithm {key!r}; known: {', '.join(REGISTRY)}")


def iter_entries(family: str | None = None) -> Iterator[Entry]:
    for entry in REGISTRY.values():
        if family is None or entry.family == family or entry.family.startswith(family):
            yield entry


def run_entry(
    key: str,
    problem: Problem,
    *,
    budget: int,
    seeds: list[int],
    sample_every: int = 25,
    **params,
) -> list[OptimizerResult]:
    """Run one registered method across several seeds."""
    entry = get(key)
    return [
        entry.build(
            problem, budget=budget, seed=seed, sample_every=sample_every, **params
        ).run()
        for seed in seeds
    ]


__all__ = [
    "REGISTRY",
    "Convergence",
    "Entry",
    "GA",
    "LocalSearchRefiner",
    "Optimizer",
    "OptimizerResult",
    "PSO",
    "QGA",
    "QISEP",
    "QISEPWithLocalSearch",
    "QPSO",
    "QPSOWithLocalSearch",
    "SavingsBaseline",
    "SimulatedAnnealing",
    "all_keys",
    "brute_force_tsp",
    "classical_keys",
    "compare_exact_searchers",
    "exact_shortest_path",
    "get",
    "held_karp",
    "iter_entries",
    "quantum_keys",
    "run_entry",
    "run_optimizer",
]
