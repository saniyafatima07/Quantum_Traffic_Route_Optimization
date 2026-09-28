"""Validate the full loop on the real Bengaluru network: optimise, then simulate.

Run with:  uv run python scripts/validate_loop.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qitransit.algorithms import REGISTRY, get
from qitransit.graph import load_net_file
from qitransit.graph.bengaluru import LANDMARKS_BY_AREA, landmark_nodes
from qitransit.problems import (
    build_travel_matrix,
    congestion_snapshot,
    instance_settings,
    make_fleet_instance,
    sample_near,
)
from qitransit.routing import is_legal_route
from qitransit.simulation import (
    SumoRunner,
    build_assignments,
    departure_schedule,
    expand_solution,
    write_routes,
)

NET_FILE = "data/networks/net/bengaluru_central.net.xml"
WORK = Path("data/simulation")


def main() -> int:
    if not Path(NET_FILE).is_file():
        print(f"missing {NET_FILE}; run: uv run qitransit bengaluru build central")
        return 1

    loaded = time.perf_counter()
    network = load_net_file(NET_FILE)
    print(f"network: {network.stats()['nodes']} nodes, {network.stats()['edges']} edges "
          f"({time.perf_counter() - loaded:.1f}s)")

    anchors = list(landmark_nodes(network, LANDMARKS_BY_AREA["central"]).values())
    nodes = sample_near(network, anchors, 31, seed=11, radius_m=3000.0)
    settings = instance_settings("bengaluru-central", len(nodes), seed=11)
    edge_cost, weights = congestion_snapshot(network, seed=5, severity=0.5)

    started = time.perf_counter()
    matrix = build_travel_matrix(network, nodes, edge_cost, weights)
    print(f"travel matrix: {len(matrix)}^2 legs in {time.perf_counter() - started:.1f}s, "
          f"mean leg {matrix.time.sum() / (len(matrix) ** 2 - len(matrix)):.0f}s")

    problem = make_fleet_instance(network, nodes, settings, edge_cost, weights, matrix)
    print("instance:", problem.describe())

    results = {}
    for key in REGISTRY:
        entry = get(key)
        t0 = time.perf_counter()
        result = entry.build(problem, budget=1500, seed=7).run()
        results[entry.name] = result
        print(f"  {entry.name:10s} obj={result.evaluation.objective:9.1f} "
              f"time={result.evaluation.time:7.1f}s veh={result.evaluation.vehicles:2d} "
              f"feasible={result.evaluation.feasible} wall={time.perf_counter() - t0:.2f}s")

    best_name = min(results, key=lambda n: results[n].evaluation.objective)
    best = results[best_name]
    print(f"best: {best_name} objective {best.evaluation.objective:.1f}")

    customer_nodes = [int(node) for node in nodes[1:]]
    legs, stops, unserved, _ = expand_solution(
        network, best.solution, problem.depot, customer_nodes, edge_cost
    )
    print(f"expanded {len(legs)} vehicle routes, {sum(len(e) for e in legs)} edges total, "
          f"{len(unserved)} unreachable customers")
    if any(not is_legal_route(network, leg) for leg in legs):
        print("ERROR: at least one expanded route is not a legal turn sequence")
        return 1

    departs = departure_schedule(len(legs), window=1800.0, seed=3)
    assignments = build_assignments(
        network, legs, stops, departs=departs, service_time=settings.service_time
    )
    routes_file = write_routes(WORK / "validated.rou.xml", assignments)
    print(f"routes written: {routes_file} ({routes_file.stat().st_size / 1024:.0f} KB)")

    runner = SumoRunner(NET_FILE, seed=42)
    report = runner.run(routes_file, steps=5400, sample_every=60)
    print("SUMO report:")
    for key, value in report.as_dict().items():
        print(f"    {key:28s} {value}")

    modelled = best.evaluation.time
    print(f"\nmodelled travel time {modelled:.0f}s vs realised {report.mean_travel_time:.0f}s "
          f"(ratio {report.mean_travel_time / max(modelled, 1e-9):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
