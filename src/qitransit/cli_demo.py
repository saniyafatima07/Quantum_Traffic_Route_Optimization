"""``qitransit demo`` — the SUMO comparison page.

The instance is assembled here rather than inside the demo package so that the
page and the benchmark harness build the *same* problem from the *same* seed: a
demo whose numbers disagreed with the report's would be worse than no demo.
"""

from __future__ import annotations

import argparse
import functools
import json
import webbrowser
from pathlib import Path

import numpy as np

from .demo.pipeline import DEFAULT_METHODS, run_comparison
from .demo.server import Builder, DemoService, serve
from .graph import load_net_file
from .graph.bengaluru import LANDMARKS_BY_AREA, landmark_nodes
from .problems import (
    build_travel_matrix,
    congestion_snapshot,
    instance_settings,
    make_fleet_instance,
    sample_near,
)

DEFAULT_NET = "data/networks/net/bengaluru_central.net.xml"


def add_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "demo",
        help="serve the SUMO comparison page (quantum-inspired vs classical)",
        description=(
            "Build one Bengaluru routing instance, solve it with every registered "
            "method on an equal budget, drive each solution through SUMO, and serve "
            "the comparison as a local web page."
        ),
    )
    parser.add_argument("--network", default=DEFAULT_NET, help="compiled .net.xml to use")
    parser.add_argument("--customers", type=int, default=30, help="customers including the depot")
    parser.add_argument("--seed", type=int, default=11, help="instance seed (customer placement)")
    parser.add_argument("--budget", type=int, default=3000, help="evaluations per method")
    parser.add_argument("--congestion-seed", type=int, default=5, help="frozen traffic snapshot seed")
    parser.add_argument(
        "--severity", type=float, default=0.5, help="congestion severity in the BPR model, 0..1"
    )
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="comma-separated registry keys")
    parser.add_argument("--steps", type=int, default=10800, help="SUMO seconds per method")
    parser.add_argument("--sumo-seed", type=int, default=42, help="SUMO's own random seed")
    parser.add_argument("--target-vehicles", type=int, default=5, help="fleet size the capacity is sized for")
    parser.add_argument("--host", default="127.0.0.1", help="address to bind")
    parser.add_argument("--port", type=int, default=8000, help="port to bind")
    parser.add_argument("--no-simulation", action="store_true", help="skip SUMO; show modelled results only")
    parser.add_argument("--cache", default="data/demo/comparison.json", help="where to cache the built comparison")
    parser.add_argument("--rebuild", action="store_true", help="ignore the cache and recompute")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    parser.add_argument("--build-only", action="store_true", help="build the comparison and exit without serving")
    return parser


def build_instance(args: argparse.Namespace) -> dict:
    """Assemble the instance and the frozen congestion snapshot."""
    net_file = Path(args.network)
    if not net_file.is_file():
        raise FileNotFoundError(
            f"missing {net_file}. Build it first with: uv run qitransit bengaluru build central"
        )
    network = load_net_file(str(net_file))
    stats = network.stats()
    anchors = list(landmark_nodes(network, LANDMARKS_BY_AREA["central"]).values())
    nodes = sample_near(network, anchors, args.customers, seed=args.seed, radius_m=3000.0)
    settings = instance_settings(
        "bengaluru-central",
        len(nodes),
        seed=args.seed,
        congestion_severity=args.severity,
        target_vehicles=args.target_vehicles,
    )
    edge_cost, weights = congestion_snapshot(
        network, seed=args.congestion_seed, severity=args.severity
    )
    matrix = build_travel_matrix(network, nodes, edge_cost, weights)
    problem = make_fleet_instance(network, nodes, settings, edge_cost, weights, matrix)
    return {
        "network": network,
        "net_file": str(net_file),
        "problem": problem,
        "customer_nodes": [int(node) for node in nodes[1:]],
        "depot": int(nodes[0]),
        "edge_cost": edge_cost,
        "stats": stats,
    }


def run(args: argparse.Namespace) -> int:
    keys = tuple(k.strip() for k in args.methods.split(",") if k.strip())
    if not keys:
        raise SystemExit("--methods selected nothing")

    prepared = functools.partial(
        build_instance,
        args=args,
    )

    def prepare() -> dict:
        built = prepared()
        return {
            "network": built["network"],
            "net_file": built["net_file"],
            "problem": built["problem"],
            "customer_nodes": built["customer_nodes"],
            "depot": built["depot"],
            "edge_cost": built["edge_cost"],
            "budget": args.budget,
            "seed": args.seed,
            "steps": args.steps,
            "sumo_seed": args.sumo_seed,
            "simulate": not args.no_simulation,
            "methods": keys,
        }

    builder = Builder(prepare)
    service = DemoService(builder, cache=args.cache)
    if args.rebuild:
        Path(args.cache).unlink(missing_ok=True)

    if args.build_only:
        def progress(stage: str, fraction: float) -> None:
            sys_print(f"[{fraction * 100:5.1f}%] {stage}")

        comparison = run_comparison(
            **builder.kwargs(),
            progress=progress,
        )
        target = comparison.save(args.cache)
        print(f"\ncomparison written to {target} ({target.stat().st_size / 1024:.0f} KB)")
        summarise(comparison)
        return 0

    httpd = serve(service, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"qitransit demo serving {url}")
    print(f"  network   {args.network}")
    print(f"  methods   {', '.join(keys)}")
    print(f"  budget    {args.budget:,} evaluations each")
    print(f"  SUMO      {args.steps:,} s per method" + ("" if args.no_simulation else ""))
    if not args.no_simulation:
        print("  the first build takes a few minutes; later starts reuse the cache")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


def summarise(comparison) -> None:
    ranked = sorted(
        (r for r in comparison.runs if r.error is None),
        key=lambda r: r.objective,
    )
    width = max((len(r.name) for r in ranked), default=10)
    # Both time columns are per vehicle, so the two sources of truth are directly
    # comparable; the last column is how far the surrogate sat from the
    # measurement, which is the number that says whether the model can be trusted.
    print(
        f"\n{'method'.ljust(width)}  {'objective':>11}  {'veh':>4}  "
        f"{'model min/veh':>13}  {'SUMO min/veh':>12}  {'delay min':>10}  {'CO2 kg':>7}  {'err %':>7}"
    )
    for run in ranked:
        sim = run.simulation or {}
        travel = sim.get("mean_travel_time_s")
        model_min = run.eval_time / run.vehicles / 60 if run.vehicles else 0.0
        if travel is None:
            print(
                f"{run.name.ljust(width)}  {run.objective:11.1f}  {run.vehicles:4d}  "
                f"{model_min:13.1f}  {'—':>12}  {'—':>10}  {'—':>7}  {'—':>7}"
            )
            continue
        error = ((travel / 60) - model_min) / model_min * 100 if model_min else 0.0
        print(
            f"{run.name.ljust(width)}  {run.objective:11.1f}  {run.vehicles:4d}  "
            f"{model_min:13.1f}  {travel / 60:12.1f}  "
            f"{sim.get('total_time_loss_s', 0) / 60:10.1f}  "
            f"{sim.get('total_co2_g', 0) / 1000:7.2f}  {error:7.1f}"
        )
    for run in comparison.runs:
        if run.error:
            print(f"{run.name}: {run.error}")


def sys_print(message: str) -> None:
    print(message, flush=True)


__all__ = ["DEFAULT_NET", "add_parser", "build_instance", "run"]
