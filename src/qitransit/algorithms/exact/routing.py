"""Exact routing baselines, wrapped so the benchmark can schedule them like any
other method.

Dijkstra is not a metaheuristic and does not iterate, so it gets its own
contract rather than being forced through the ``Optimizer`` interface: the
benchmark records its wall-clock time and its solution, and the report compares
that time against the time the metaheuristics need to reach the same quality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ...graph.network import RoadNetwork
from ...problems.base import Evaluation, Problem
from ...routing.shortest_path import a_star, bidirectional_dijkstra, dijkstra


@dataclass(slots=True)
class ExactRouteResult:
    """An exact single-path solution and the time it took to certify."""

    evaluation: Evaluation
    wall_time: float
    edges: tuple[int, ...] = ()
    nodes: tuple[int, ...] = ()
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "objective": self.evaluation.objective,
            "time_s": round(self.evaluation.time, 3),
            "distance_m": round(self.evaluation.distance, 2),
            "edges": len(self.edges),
            "wall_time_s": round(self.wall_time, 6),
            **self.meta,
        }


def exact_shortest_path(
    network: RoadNetwork, cost: np.ndarray, source: int, target: int
) -> ExactRouteResult:
    """Dijkstra's optimal path, with the congestion objective applied afterwards."""
    started = time.perf_counter()
    result = dijkstra(network, cost, source, target)
    wall_time = time.perf_counter() - started
    lengths = network.lengths()
    free_flow = network.free_flow_time()
    edge_list = list(result.edges)
    travel = float(cost[edge_list].sum()) if edge_list else 0.0
    evaluation = Evaluation(
        objective=travel,
        time=travel,
        distance=float(lengths[edge_list].sum()) if edge_list else 0.0,
        externality=travel - float(free_flow[edge_list].sum()) if edge_list else 0.0,
        vehicles=1,
        feasible=bool(result.reached),
    )
    return ExactRouteResult(
        evaluation=evaluation,
        wall_time=wall_time,
        edges=result.edges,
        nodes=result.nodes,
        meta={"algorithm": "dijkstra"},
    )


def compare_exact_searchers(
    network: RoadNetwork, cost: np.ndarray, source: int, target: int
) -> dict[str, ExactRouteResult]:
    """All three exact searchers on the same query, for the report's timing table."""
    results: dict[str, ExactRouteResult] = {}
    for name, function in (
        ("dijkstra", dijkstra),
        ("a-star", a_star),
        ("bidirectional", bidirectional_dijkstra),
    ):
        started = time.perf_counter()
        found = function(network, cost, source, target)
        wall_time = time.perf_counter() - started
        lengths = network.lengths()
        free_flow = network.free_flow_time()
        edge_list = list(found.edges)
        travel = float(cost[edge_list].sum()) if edge_list else 0.0
        results[name] = ExactRouteResult(
            evaluation=Evaluation(
                objective=travel,
                time=travel,
                distance=float(lengths[edge_list].sum()) if edge_list else 0.0,
                externality=travel - float(free_flow[edge_list].sum()) if edge_list else 0.0,
                vehicles=1,
                feasible=bool(found.reached),
            ),
            wall_time=wall_time,
            edges=found.edges,
            nodes=found.nodes,
            meta={"algorithm": name},
        )
    return results


def certify_against_dijkstra(problem: Problem, solution) -> dict:
    """Check a proposed solution against the exact cost of each of its legs.

    A metaheuristic's reported objective is only meaningful if it agrees with the
    exact cost of the routes it claims to be driving, so this is the check the
    benchmark runs before a result is written down.
    """
    evaluation = problem.evaluate(solution)
    matrix = getattr(problem, "matrix", None)
    if matrix is None:
        return {"checked": False}
    total = 0.0
    for route in solution:
        previous = 0
        for customer in route:
            total += float(matrix.cost[previous, customer + 1])
            previous = customer + 1
        total += float(matrix.cost[previous, 0])
    reported = evaluation.objective - evaluation.penalty
    return {
        "checked": True,
        "leg_cost": total,
        "reported_unpenalised": reported,
        "agrees": bool(np.isclose(total, reported, rtol=1e-6, atol=1e-6)),
        "deviation": abs(total - reported),
    }
