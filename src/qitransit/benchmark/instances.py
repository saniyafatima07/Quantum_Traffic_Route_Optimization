"""Instance suites: the problems a benchmark sweep runs over.

Two families are on offer and they answer different questions.

The **synthetic** suites (a Manhattan grid, its larger siblings) are the ones to
draw *algorithmic* conclusions from: instances are cheap to regenerate, the
optimum is computable by Held–Karp or by exhaustive search, and the exact gap to
optimum can therefore be stated rather than approximated.

The **Bengaluru** suites are the ones to draw *practical* conclusions from: real
geometry, real one-way systems, real turn restrictions, and leg times of the
order of ten minutes.  Their optima are not computable, so they are scored
against the best solution any method found, and every claim from them is stated
relative to that rather than to an optimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..graph.network import RoadNetwork
from ..problems import (
    CongestedShortestPath,
    FleetVRP,
    TravelingSalesman,
    congestion_snapshot,
    instance_settings,
    make_fleet_instance,
    make_spp_instance,
    make_tsp_instance,
    sample_near,
    sample_nodes,
)
from ..problems.base import Problem
from ..graph.bengaluru import LANDMARKS_BY_AREA, landmark_nodes

DEFAULT_NET = Path("data/networks/net/bengaluru_central.net.xml")


@dataclass(frozen=True, slots=True)
class SuitePlan:
    """How to build one family of instances."""

    name: str
    problem_family: str
    sizes: tuple[int, ...]
    seeds: tuple[int, ...]
    time_windows: bool = False
    target_vehicles: int = 5
    n_vehicles: int = 12

    def describe(self) -> str:
        tw = " +time windows" if self.time_windows else ""
        return (
            f"{self.name}: {len(self.sizes)} sizes {list(self.sizes)} "
            f"x {len(self.seeds)} seeds{tw}"
        )


BENGALURU_VRP = SuitePlan("vrp", "cvrp", (20, 40, 80), (1, 2, 3, 4, 5), target_vehicles=5)
BENGALURU_VRPTW = SuitePlan("vrptw", "cvrp", (20, 40, 80), (1, 2, 3, 4, 5), time_windows=True)
BENGALURU_TSP = SuitePlan("tsp", "tsp", (12, 18, 25), (1, 2, 3, 4, 5))
BENGALURU_SPP = SuitePlan("spp", "spp", (10, 20, 40), (1, 2, 3, 4, 5), n_vehicles=12)

PLANS: dict[str, SuitePlan] = {
    p.name: p for p in (BENGALURU_VRP, BENGALURU_VRPTW, BENGALURU_TSP, BENGALURU_SPP)
}


def load_network(net_file: str | Path = DEFAULT_NET) -> RoadNetwork:
    from ..graph import load_net_file

    path = Path(net_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; build it with: uv run qitransit network bengaluru build central"
        )
    return load_net_file(path)


def demand_nodes(network: RoadNetwork, count: int, seed: int, clustered: bool) -> np.ndarray:
    """Depot plus customers, clustered around real landmarks when asked.

    Uniform random junctions make a poor proxy for delivery demand, which is
    concentrated along commercial corridors; clustering around the area's named
    landmarks is what gives the congestion term something to bite on.
    """
    if not clustered:
        return sample_nodes(network, count, seed)
    anchors = list(landmark_nodes(network, LANDMARKS_BY_AREA["central"]).values())
    return sample_near(network, anchors, count, seed, radius_m=3000.0)


def build_problems(
    plan: SuitePlan,
    network: RoadNetwork,
    *,
    clustered: bool = True,
    congestion_seed: int = 5,
    severity: float = 0.5,
    progress: bool = True,
) -> list[tuple[str, Problem]]:
    """Every instance of a plan, in a deterministic order.

    The congestion snapshot is built once and shared: a different rush hour per
    instance would be more realistic and would destroy the comparison.
    """
    edge_cost, weights = congestion_snapshot(network, seed=congestion_seed, severity=severity)
    problems: list[tuple[str, Problem]] = []
    for size in plan.sizes:
        for seed in plan.seeds:
            nodes = demand_nodes(network, size, seed, clustered)
            settings = instance_settings(
                "bengaluru-central",
                int(nodes.size),
                seed,
                use_time_windows=plan.time_windows,
                target_vehicles=plan.target_vehicles,
            )
            name = settings.name
            if progress:
                print(f"  building {name}", flush=True)
            if plan.problem_family == "tsp":
                problem: Problem = make_tsp_instance(
                    network, nodes, settings, edge_cost, weights
                )
            elif plan.problem_family == "spp":
                problem = make_spp_instance(
                    network,
                    int(nodes[0]),
                    nodes[1:],
                    settings,
                    edge_cost,
                    n_vehicles=plan.n_vehicles,
                )
            else:
                problem = make_fleet_instance(network, nodes, settings, edge_cost, weights)
            problems.append((name, problem))
    return problems


def synthetic_problems(
    plan: SuitePlan,
    preset_name: str = "manhattan",
    *,
    rows: int = 12,
    cols: int = 12,
    progress: bool = True,
) -> list[tuple[str, Problem]]:
    """Small instances on a grid, where the exact optimum is computable."""
    from ..graph.builder import build_city, preset

    network = build_city(preset(preset_name, rows=rows, cols=cols))
    edge_cost, weights = congestion_snapshot(network, seed=5, severity=0.4)
    problems: list[tuple[str, Problem]] = []
    for size in plan.sizes:
        for seed in plan.seeds:
            nodes = sample_nodes(network, size, seed, min_degree=3)
            settings = instance_settings(
                preset_name, int(nodes.size), seed, target_vehicles=plan.target_vehicles
            )
            if progress:
                print(f"  building {settings.name}", flush=True)
            if plan.problem_family == "tsp":
                problem: Problem = make_tsp_instance(
                    network, nodes, settings, edge_cost, weights
                )
            else:
                problem = make_fleet_instance(network, nodes, settings, edge_cost, weights)
            problems.append((settings.name, problem))
    return problems
