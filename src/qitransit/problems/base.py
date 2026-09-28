"""Shared problem vocabulary: solutions, evaluations and the optimiser contract.

Every routing problem in this package exposes the *same* interface to the
optimisers, and every optimiser works on the *same* encoding: a real-valued
priority vector with one entry per customer.  That is what makes the benchmark
fair — QPSO, PSO, GA and SA differ only in how they move the vector, never in
what they are allowed to see.

The decoding pipeline is always:

    priority vector -> visit order -> feasible routes -> objective

so an algorithm that proposes a better vector proposes a better set of routes,
regardless of whether the vector came from a quantum-inspired update or a
classical one.
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

Route = tuple[int, ...]
Solution = tuple[Route, ...]


@dataclass(frozen=True, slots=True)
class Evaluation:
    """What a solution scores.

    ``objective`` is the value optimisers minimise: the weighted sum of the
    routing criteria plus the constraint penalty.  The remaining fields are the
    unpenalised breakdown used in reports, so a reader can always see whether a
    good objective came from real savings or from tolerating violations.
    """

    objective: float
    time: float
    distance: float
    externality: float
    vehicles: int
    penalty: float = 0.0
    capacity_violation: float = 0.0
    time_window_violation: float = 0.0
    unserved: int = 0
    feasible: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Customer:
    """A delivery stop."""

    node: int
    demand: float = 0.0
    service_time: float = 120.0
    window_start: float = 0.0
    window_end: float = 86400.0
    name: str = ""

    @property
    def window_width(self) -> float:
        return self.window_end - self.window_start


@dataclass
class ProblemSpec:
    """Instance metadata carried alongside a problem for reproducible reports."""

    name: str
    family: str
    n_customers: int
    n_vehicles: int
    seed: int
    capacity: float
    horizon: float
    network: str
    notes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class Problem(abc.ABC):
    """A decodable optimisation problem.

    Subclasses fix the feasibility rules and the objective; the decoding of a
    priority vector into a feasible solution is the part every optimiser shares.
    """

    family: str = "abstract"

    def __init__(self, name: str, n_customers: int, seed: int = 0) -> None:
        self.name = name
        self.n_customers = n_customers
        self.seed = seed

    # -- the interface the optimisers use ---------------------------
    @property
    def size(self) -> int:
        """Dimensionality of the priority vector."""
        return self.n_customers

    @abc.abstractmethod
    def decode(self, keys: np.ndarray) -> Solution:
        """Turn a priority vector into a feasible solution."""

    @abc.abstractmethod
    def evaluate(self, solution: Solution) -> Evaluation:
        """Score a solution."""

    @abc.abstractmethod
    def order_of(self, keys: np.ndarray) -> tuple[int, ...]:
        """The visit order a priority vector implies, before route splitting."""

    @abc.abstractmethod
    def encode(self, solution: Solution) -> np.ndarray:
        """Inverse of :meth:`order_of`, for memetic operators that start from a solution."""

    def random_keys(self, rng: np.random.Generator) -> np.ndarray:
        """A random starting point for the swarm or population."""
        return rng.random(self.size)

    def lower_bound(self) -> float:
        """A valid lower bound on the objective, or ``-inf`` when unknown."""
        return float("-inf")

    def spec(self) -> ProblemSpec:
        return ProblemSpec(
            name=self.name,
            family=self.family,
            n_customers=self.n_customers,
            n_vehicles=getattr(self, "n_vehicles", 0),
            seed=self.seed,
            capacity=float(getattr(self, "capacity", 0.0)),
            horizon=float(getattr(self, "horizon", 0.0)),
            network=getattr(self, "network_name", "synthetic"),
            notes=dict(getattr(self, "notes", {})),
        )

    def evaluate_keys(self, keys: np.ndarray) -> Evaluation:
        return self.evaluate(self.decode(keys))

    # -- helpers -----------------------------------------------------
    def all_customers(self) -> set[int]:
        return set(range(self.n_customers))

    def solved(self, solution: Solution) -> bool:
        served = {c for route in solution for c in route}
        return served == self.all_customers()

    def is_feasible(self, solution: Solution) -> bool:
        return self.evaluate(solution).feasible

    def as_permutation(self, solution: Solution) -> tuple[int, ...]:
        flat: list[int] = []
        for route in solution:
            flat.extend(route)
        return tuple(flat)


def keys_from_order(order: Sequence[int], n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Priority vector reproducing a given visit order."""
    keys = np.empty(n, dtype=np.float64)
    if rng is None:
        keys = np.linspace(1.0, 0.0, n, endpoint=False)
    else:
        keys = rng.random(n)
    rank = np.empty(n, dtype=np.int64)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(len(order))
    return keys + 0.001 * rank


def flatten_routes(solution: Solution) -> list[int]:
    return [c for route in solution for c in route]


def route_demands(solution: Solution, demands: np.ndarray) -> np.ndarray:
    return np.array(
        [sum(demands[c] for c in route) for route in solution], dtype=np.float64
    )
