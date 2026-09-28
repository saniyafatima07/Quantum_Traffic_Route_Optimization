"""The ``qitransit`` command line.

Five verbs, in the order a newcomer meets them:

``bengaluru``
    Get a network.  Extract Bengaluru from OpenStreetMap, or build a small
    synthetic city to test on in a second.
``methods``
    See what is registered, and what each one takes.
``solve``
    Run one method on one instance and print the answer.
``simulate``
    Take a routes file and drive it through SUMO.
``demo``
    The comparison page: every method, every route simulated, side by side.

``demo`` is the one to reach for.  Everything else exists to produce the pieces
it puts together, and to check those pieces independently.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_NET = "data/networks/net/bengaluru_central.net.xml"


def add_bengaluru(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "bengaluru",
        help="extract and compile the real Bengaluru network from OpenStreetMap",
    )
    actions = parser.add_subparsers(dest="action", required=True)

    extract = actions.add_parser("extract", help="clip the Geofabrik extract to an area")
    extract.add_argument("--area", default="central", choices=["central", "city", "metro"])
    extract.add_argument("--out", default=None, help="write the clipped .osm.pbf here")
    extract.add_argument("--force", action="store_true", help="re-clip even if the output exists")

    build = actions.add_parser("build", help="compile an area into a SUMO .net.xml")
    build.add_argument("--area", default="central", choices=["central", "city", "metro"])
    build.add_argument("--out", default=DEFAULT_NET)
    build.add_argument("--seed", type=int, default=42, help="netconvert randomness")
    build.add_argument("--force", action="store_true", help="recompile even if the output exists")

    return parser


def add_methods(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("methods", help="list the registered optimisers")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    return parser


def add_solve(subparsers) -> argparse.ArgumentParser:
    from .cli_demo import DEFAULT_NET as net_default

    parser = subparsers.add_parser("solve", help="run one method on one instance")
    parser.add_argument("method", help="registry key, e.g. qpso, qisep-ls, sa")
    parser.add_argument("--network", default=net_default)
    parser.add_argument("--customers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument("--target-vehicles", type=int, default=5)
    parser.add_argument("--routes-out", default=None, help="write an expanded .rou.xml here")
    parser.add_argument("--json", action="store_true")
    return parser


def add_simulate(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("simulate", help="drive a routes file through SUMO")
    parser.add_argument("routes", help="a .rou.xml with explicit edge routes")
    parser.add_argument("--network", default=DEFAULT_NET)
    parser.add_argument("--steps", type=int, default=10800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true", help="open sumo-gui instead of running headless")
    parser.add_argument("--json", action="store_true")
    return parser


def add_net(subparsers) -> argparse.ArgumentParser:
    from .graph import PRESETS

    parser = subparsers.add_parser("net", help="build a small synthetic city, for testing")
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build", help="compile a preset")
    build.add_argument("preset", choices=sorted(PRESETS))
    build.add_argument("--out", default=None)
    build.add_argument("--seed", type=int, default=1)
    info = actions.add_parser("info", help="print a compiled network's statistics")
    info.add_argument("path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qitransit",
        description="Quantum-inspired routing metaheuristics on real urban networks, validated with SUMO.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_bengaluru(subparsers)
    add_net(subparsers)
    add_methods(subparsers)
    add_solve(subparsers)
    add_simulate(subparsers)

    from .cli_demo import add_parser as add_demo

    add_demo(subparsers)

    args = parser.parse_args(argv)
    handler = {
        "bengaluru": _bengaluru,
        "net": _net,
        "methods": _methods,
        "solve": _solve,
        "simulate": _simulate,
        "demo": _demo,
    }[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _bengaluru(args) -> int:
    from .graph.bengaluru import AREAS, build, extract_osm

    if args.action == "extract":
        target = Path(args.out) if args.out else Path("data/networks/raw") / f"bengaluru_{args.area}.osm.pbf"
        if target.is_file() and not args.force:
            print(f"{target} already exists ({target.stat().st_size / 1e6:.0f} MB); use --force to re-clip")
            return 0
        spec = AREAS[args.area]
        print(f"clipping {spec.source} to {args.area} ({spec.label})…")
        extract_osm(spec, target, force=args.force)
        print(f"wrote {target} ({target.stat().st_size / 1e6:.0f} MB)")
        return 0

    target = Path(args.out)
    if target.is_file() and not args.force:
        print(f"{target} already exists; use --force to recompile")
        return 0
    print(f"compiling {args.area}…")
    started = time.perf_counter()
    result = build(args.area, target, seed=args.seed, force=args.force)
    print(f"wrote {result} ({time.perf_counter() - started:.0f}s)")
    return 0


def _net(args) -> int:
    from .graph import build_preset, load_net_file

    if args.action == "build":
        target = Path(args.out) if args.out else Path("data/networks/net") / f"{args.preset}.net.xml"
        print(f"building preset {args.preset}…")
        result = build_preset(args.preset, target, seed=args.seed)
        print(f"wrote {result}")
        return 0

    network = load_net_file(args.path)
    print(json.dumps(network.stats(), indent=2))
    return 0


def _methods(args) -> int:
    from .algorithms import REGISTRY

    if args.json:
        print(json.dumps({k: {"name": e.name, "family": e.family} for k, e in REGISTRY.items()}, indent=2))
        return 0
    width = max(len(k) for k in REGISTRY)
    print(f"{'key'.ljust(width)}  {'family':<16}  name")
    for key, entry in REGISTRY.items():
        print(f"{key.ljust(width)}  {entry.family:<16}  {entry.name}")
    return 0


def _solve(args) -> int:
    import numpy as np

    from .algorithms import get
    from .simulation import build_assignments, departure_schedule, expand_solution, write_routes
    from .cli_demo import build_instance

    built = build_instance(args)
    network = built["network"]
    problem = built["problem"]
    entry = get(args.method)
    print(f"instance: {problem.describe()}")
    started = time.perf_counter()
    result = entry.build(problem, budget=args.budget, seed=args.seed).run()
    wall = time.perf_counter() - started
    evaluation = result.evaluation
    payload = {
        "method": entry.name,
        "objective": round(evaluation.objective, 3),
        "feasible": evaluation.feasible,
        "vehicles": evaluation.vehicles,
        "unserved": evaluation.unserved,
        "time_s": round(evaluation.time, 1),
        "distance_m": round(evaluation.distance, 1),
        "penalty": round(evaluation.penalty, 3),
        "evaluations": result.n_evals,
        "wall_time_s": round(wall, 3),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for key, value in payload.items():
            print(f"  {key:<16} {value}")

    if args.routes_out:
        legs, stops, unserved, _ = expand_solution(
            network,
            result.solution,
            built["depot"],
            built["customer_nodes"],
            built["edge_cost"],
        )
        routes = write_routes(
            Path(args.routes_out),
            build_assignments(
                network,
                legs,
                stops,
                departs=departure_schedule(len(legs), seed=args.seed),
            ),
        )
        print(f"  routes          {routes} ({len(legs)} vehicles, {sum(len(l) for l in legs)} edges, {len(unserved)} unreachable)")
    return 0


def _simulate(args) -> int:
    from .simulation import SumoRunner
    from .sumo_env import _executable

    routes = Path(args.routes)
    if not routes.is_file():
        raise FileNotFoundError(f"no routes file at {routes}")

    if args.gui:
        import subprocess

        subprocess.Popen(
            [_executable("sumo-gui"), "-n", args.network, "-r", str(routes), "--start"]
        )
        return 0

    report = SumoRunner(args.network, seed=args.seed).run(
        routes, steps=args.steps, sample_every=60
    )
    payload = report.as_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for key, value in payload.items():
            print(f"  {key:<28} {value}")
    return 0


def _demo(args) -> int:
    from .cli_demo import run

    return run(args)


__all__ = ["main"]
