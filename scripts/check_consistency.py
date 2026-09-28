"""Invariants the demo depends on, checked against every registered method.

The comparison page puts a modelled objective and a SUMO measurement next to each
other for the same row, so the objective has to describe the routes that were
actually written out.  These checks fail loudly if that ever stops being true.

Run:  uv run python scripts/check_consistency.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from qitransit.algorithms import REGISTRY, get
from qitransit.cli_demo import build_instance
from qitransit.routing import is_legal_route
from qitransit.simulation.sumo import expand_solution


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network", default="data/networks/net/bengaluru_central.net.xml")
    p.add_argument("--customers", type=int, default=30)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--congestion-seed", type=int, default=1)
    p.add_argument("--severity", type=float, default=0.5)
    p.add_argument("--target-vehicles", type=int, default=5)
    p.add_argument("--budget", type=int, default=1500)
    p.add_argument("--methods", default=",".join(sorted(REGISTRY)))
    return p


def main() -> int:
    args = parser().parse_args()
    built = build_instance(args)
    network = built["network"]
    problem = built["problem"]
    lengths = network.lengths()
    keys = [k.strip() for k in args.methods.split(",") if k.strip()]

    print(
        f"{'method':<10} {'objective':>10} {'re-scored':>10} {'match':>6} "
        f"{'veh':>4} {'served':>7} {'legal':>6} {'priced km':>10} {'driven km':>10} {'driven/priced':>13}"
    )
    failures: list[str] = []

    for key in keys:
        result = get(key).build(problem, budget=args.budget, seed=args.seed, sample_every=25).run()
        solution = result.solution
        reported = result.evaluation
        rescored = problem.evaluate(solution)

        if abs(rescored.objective - reported.objective) > 1e-9:
            failures.append(
                f"{key}: reported objective {reported.objective:.1f} does not describe its own "
                f"solution, which re-scores to {rescored.objective:.1f}"
            )

        served = sum(len(route) for route in solution)
        if served != problem.n_customers:
            failures.append(f"{key}: served {served} of {problem.n_customers} customers")
        if reported.unserved:
            failures.append(f"{key}: {reported.unserved} customers left unserved")

        legs, _, _, _ = expand_solution(
            network, solution, built["depot"], built["customer_nodes"], built["edge_cost"]
        )
        illegal = [i for i, leg in enumerate(legs) if not is_legal_route(network, leg)]
        if illegal:
            failures.append(f"{key}: routes {illegal} contain an illegal turn")

        priced = 0.0
        for route in solution:
            previous = 0
            for customer in route:
                nxt = customer + 1
                priced += float(problem.matrix.distance[previous, nxt])
                previous = nxt
            priced += float(problem.matrix.distance[previous, 0])
        driven = sum(sum(float(lengths[e]) for e in leg) for leg in legs)
        ratio = driven / priced if priced else 0.0
        if not np.isfinite(ratio) or not 0.5 < ratio < 1.5:
            failures.append(f"{key}: driven/priced distance ratio {ratio:.3f} is not plausible")

        print(
            f"{key:<10} {reported.objective:10.1f} {rescored.objective:10.1f} "
            f"{'ok' if abs(rescored.objective - reported.objective) <= 1e-9 else 'NO':>6} "
            f"{reported.vehicles:4d} {served:7d} {len(legs) - len(illegal):6d} "
            f"{priced / 1000:10.2f} {driven / 1000:10.2f} {ratio:13.3f}"
        )

    if failures:
        print("\nFAILED:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print(f"\nall {len(keys)} methods consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
