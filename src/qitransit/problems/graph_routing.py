"""Routing problems solved by the platform.

* :class:`FleetVRP` — capacitated / time-windowed fleet routing.
* :class:`CongestedShortestPath` — a *fleet* of vehicles sharing a congestion
  charge, where the optimum is genuinely not a shortest path.
* :class:`TravelingSalesman` — the unbounded special case, kept for
  cross-checking against the exact Held-Karp solver.
"""

from __future__ import annotations

import numpy as np

from ..graph.network import RoadNetwork
from ..routing.shortest_path import dijkstra, dijkstra_tree
from .base import Evaluation, Problem, Solution
from .travel import TravelMatrix


class TravelingSalesman(Problem):
    """Single-vehicle tour over all customers from the depot."""

    family = "tsp"

    def __init__(
        self,
        matrix: TravelMatrix,
        *,
        name: str = "tsp",
        seed: int = 0,
        weights: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        network_name: str = "synthetic",
        notes: dict | None = None,
    ) -> None:
        super().__init__(name=name, n_customers=len(matrix) - 1, seed=seed)
        self.matrix = matrix
        self.w_time, self.w_dist, self.w_cong, self.w_veh = weights
        self.network_name = network_name
        self.notes = dict(notes or {})

    def order_of(self, keys: np.ndarray) -> tuple[int, ...]:
        return tuple(int(i) - 1 for i in np.argsort(-np.asarray(keys), kind="stable"))

    def encode(self, solution: Solution) -> np.ndarray:
        order = self.as_permutation(solution)
        n = self.n_customers
        if not order:
            return np.zeros(n)
        base = np.linspace(1.0, 0.0, n, endpoint=False)
        keys = np.empty(n, dtype=np.float64)
        for rank, customer in enumerate(order):
            keys[customer] = base[rank]
        return keys

    def decode(self, keys: np.ndarray) -> Solution:
        return (self.order_of(keys),)

    def evaluate(self, solution: Solution) -> Evaluation:
        if not solution or not solution[0]:
            return Evaluation(0.0, 0.0, 0.0, 0.0, 1, feasible=False, unserved=self.n_customers)
        route = solution[0]
        previous = 0
        total_time = total_dist = total_ext = 0.0
        for customer in route:
            nxt = customer + 1
            total_time += float(self.matrix.time[previous, nxt])
            total_dist += float(self.matrix.distance[previous, nxt])
            total_ext += float(self.matrix.externality[previous, nxt])
            previous = nxt
        total_time += float(self.matrix.time[previous, 0])
        total_dist += float(self.matrix.distance[previous, 0])
        total_ext += float(self.matrix.externality[previous, 0])
        objective = (
            self.w_time * total_time
            + self.w_dist * total_dist
            + self.w_cong * total_ext
        )
        served = len(set(route))
        return Evaluation(
            objective=objective,
            time=total_time,
            distance=total_dist,
            externality=total_ext,
            vehicles=1,
            unserved=self.n_customers - served,
            feasible=served == self.n_customers,
        )

    def lower_bound(self) -> float:
        if self.n_customers == 0:
            return 0.0
        inbound = self.matrix.cost[1:, 0]
        outbound = self.matrix.cost[0, 1:]
        mask = np.isfinite(inbound) & np.isfinite(outbound)
        if not mask.any():
            return float("-inf")
        return float(self.w_time * np.minimum(inbound, outbound)[mask].sum())


class CongestedShortestPath(Problem):
    """``k`` vehicles from a hub to customers under a shared congestion charge.

    With a single vehicle the optimum is a shortest path and Dijkstra solves it,
    which is exactly the point of the benchmark: the exact method is included so
    the report can show where the metaheuristic loses nothing and where the
    extra structure starts to pay.

    The congestion charge is the classic Pigou externality: every vehicle that
    uses an edge slows down the others, so the cost of a leg rises with the
    number of vehicles routed over it.  Optimising jointly is a
    system-optimal assignment problem, not ``k`` independent shortest paths.
    """

    family = "spp"

    def __init__(
        self,
        network: RoadNetwork,
        source: int,
        targets: list[int],
        edge_cost: np.ndarray,
        *,
        n_vehicles: int = 1,
        congestion_strength: float = 0.35,
        name: str = "spp",
        seed: int = 0,
        network_name: str = "synthetic",
        notes: dict | None = None,
    ) -> None:
        super().__init__(name=name, n_customers=len(targets), seed=seed)
        self.network = network
        self.source = source
        self.targets = list(targets)
        self.n_vehicles = n_vehicles
        self.congestion_strength = congestion_strength
        self.edge_cost = np.asarray(edge_cost, dtype=np.float64)
        self.network_name = network_name
        self.notes = dict(notes or {})
        self.free_flow = network.free_flow_time()
        self.lengths = network.lengths()

        dist, prev_node, prev_edge = dijkstra_tree(network, self.edge_cost, source)
        self.baseline_cost = np.array([dist[t] for t in self.targets], dtype=np.float64)

    def order_of(self, keys: np.ndarray) -> tuple[int, ...]:
        """Targets in descending priority, the sequence vehicles are dispatched."""
        return tuple(int(i) for i in np.argsort(-np.asarray(keys), kind="stable"))

    def encode(self, solution: Solution) -> np.ndarray:
        order = self.as_permutation(solution)
        n = self.n_customers
        if not order:
            return np.zeros(n)
        base = np.linspace(1.0, 0.0, n, endpoint=False)
        keys = np.empty(n, dtype=np.float64)
        for rank, target in enumerate(order):
            keys[target] = base[rank]
        return keys

    def decode(self, keys: np.ndarray) -> Solution:
        order = self.order_of(keys)
        return (order,) if order else ()

    def leg_edges(self, target_index: int) -> tuple[int, ...]:
        result = dijkstra(self.network, self.edge_cost, self.source, self.targets[target_index])
        return result.edges

    def evaluate(self, solution: Solution) -> Evaluation:
        if not solution or not solution[0]:
            return Evaluation(0.0, 0.0, 0.0, 0.0, 0, feasible=False, unserved=self.n_customers)
        order = solution[0]
        edges: list[int] = []
        for target_index in order:
            edges.extend(self.leg_edges(target_index))

        if not edges:
            return Evaluation(0.0, 0.0, 0.0, 0.0, 0, feasible=False, unserved=self.n_customers)

        usage = np.bincount(edges, minlength=self.network.n_edges).astype(np.float64)
        vehicles = float(max(self.n_vehicles, 1))
        load = np.clip(usage / vehicles, 0.0, 1.5)
        penalty = 1.0 + self.congestion_strength * load**2
        charged = self.edge_cost * penalty

        total_cost = float(charged[edges].sum())
        total_time = float(self.free_flow[edges].sum())
        total_dist = float(self.lengths[edges].sum())
        externality = float((charged[edges] - self.edge_cost[edges]).sum())
        served = len(set(order))
        return Evaluation(
            objective=total_cost,
            time=total_time,
            distance=total_dist,
            externality=externality,
            vehicles=self.n_vehicles,
            unserved=self.n_customers - served,
            feasible=served == self.n_customers,
        )

    def lower_bound(self) -> float:
        reachable = self.baseline_cost[np.isfinite(self.baseline_cost)]
        if reachable.size == 0:
            return float("-inf")
        return float(reachable.sum())
