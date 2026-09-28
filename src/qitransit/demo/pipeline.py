"""The comparison the demo shows: quantum-inspired against classical, on SUMO.

The pipeline is deliberately the same one the project uses for its own results,
because a demo that took a different path would be measuring something other
than what it claims:

1.  Build one instance and one frozen congestion snapshot, shared by everyone.
2.  Give every method the *same* evaluation budget and the same seed.
3.  Take each method's best solution and turn it into a drivable edge route.
4.  Run each of those routes through SUMO against the real network.
5.  Report both what the optimiser believed and what the simulator measured.

Step 5 is the part that matters.  The optimiser scores routes against a *BPR*
congestion model — a closed-form function of flow — while SUMO resolves vehicles
one at a time against real junctions, real signals and real lane discipline.  A
method can win the model and lose the road, and only running the second can tell
the two apart.  So every number the UI shows in the *measured* column comes from
TraCI, not from the objective function.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..algorithms import REGISTRY, get
from ..graph.network import RoadNetwork
from ..problems.base import Problem
from ..simulation import (
    SumoRunner,
    build_assignments,
    departure_schedule,
    expand_solution,
    write_routes,
)

#: Each quantum-inspired method against the classical algorithm it is a variant
#: of, plus the construction heuristic that needs no search at all.
DEFAULT_METHODS: tuple[str, ...] = (
    "qpso",
    "qpso-ls",
    "qisep",
    "qisep-ls",
    "qga",
    "pso",
    "ga",
    "sa",
    "savings",
)

Progress = Callable[[str, float], None]


def _noop(stage: str, fraction: float) -> None:
    return None


@dataclass
class MethodRun:
    """One method's solution, plus what the simulator made of it."""

    key: str
    name: str
    family: str
    objective: float
    feasible: bool
    vehicles: int
    unserved: int
    penalty: float
    eval_time: float
    eval_distance: float
    eval_externality: float
    n_evals: int
    wall_time: float
    trace_evals: list[int] = field(default_factory=list)
    trace_costs: list[float] = field(default_factory=list)
    polylines: list[list[list[float]]] = field(default_factory=list)
    waypoints: list[list[dict]] = field(default_factory=list)
    routes_file: str = ""
    simulation: dict | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "family": self.family,
            "objective": round(self.objective, 2),
            "feasible": self.feasible,
            "vehicles": self.vehicles,
            "unserved": self.unserved,
            "penalty": round(self.penalty, 3),
            "eval_time": round(self.eval_time, 1),
            "eval_distance": round(self.eval_distance, 1),
            "eval_externality": round(self.eval_externality, 1),
            "n_evals": self.n_evals,
            "wall_time": round(self.wall_time, 3),
            "trace_evals": self.trace_evals,
            "trace_costs": self.trace_costs,
            "polylines": self.polylines,
            "waypoints": self.waypoints,
            "routes_file": self.routes_file,
            "simulation": self.simulation,
            "error": self.error,
        }


@dataclass
class Comparison:
    """The whole demo: instance, methods, network geometry, measurement."""

    meta: dict
    basemap: list[list[float]]
    runs: list[MethodRun] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "meta": self.meta,
            "basemap": self.basemap,
            "runs": [r.as_dict() for r in self.runs],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), separators=(",", ":")), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> Comparison:
        """Read a cached comparison back.

        ``as_dict`` emits exactly the dataclass fields, so a saved comparison
        reopens into an equal object.  A mismatch is raised rather than
        patched: the alternative is a cache that looks present and quietly
        serves numbers that came from nowhere.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = MethodRun.__dataclass_fields__
        return cls(
            meta=payload["meta"],
            basemap=payload["basemap"],
            runs=[MethodRun(**{k: v for k, v in row.items() if k in fields}) for row in payload["runs"]],
        )


#: Road classes worth drawing, and how prominent each is.  Anything absent is
#: left out: the full network is 156k edges, and at city scale the service
#: lanes are sub-pixel.  The arterial skeleton carries the shape; the routes
#: carry the information.
BASEMAP_RANK: dict[str, int] = {
    "highway.motorway": 0,
    "highway.trunk": 0,
    "highway.trunk_link": 1,
    "highway.primary": 1,
    "highway.primary_link": 2,
    "highway.secondary": 2,
    "highway.secondary_link": 3,
    "highway.tertiary": 3,
    "highway.tertiary_link": 4,
}


def basemap(network: RoadNetwork, max_rank: int = 3) -> list[list[float]]:
    """Every road above a rank, flattened for a canvas draw."""
    coords = network.coordinates()
    rows: list[list[float]] = []
    for edge in network.edges:
        rank = BASEMAP_RANK.get(edge.road_class, 9)
        if rank > max_rank:
            continue
        x1, y1 = coords[edge.from_node]
        x2, y2 = coords[edge.to_node]
        rows.append([round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1), rank])
    return rows


def polylines(network: RoadNetwork, legs: Sequence[Sequence[int]]) -> list[list[list[float]]]:
    """Node geometry for each vehicle's edge sequence.

    Consecutive edges share a junction, so the polyline is the tail of the first
    edge followed by every edge's head — repeating that shared point would
    double the payload for nothing.
    """
    coords = network.coordinates()
    out: list[list[list[float]]] = []
    for edges in legs:
        if not edges:
            out.append([])
            continue
        first = network.edges[edges[0]]
        points = [
            [round(float(coords[first.from_node][0]), 1), round(float(coords[first.from_node][1]), 1)]
        ]
        for edge_index in edges:
            head = network.edges[edge_index].to_node
            points.append([round(float(coords[head][0]), 1), round(float(coords[head][1]), 1)])
        out.append(points)
    return out


def waypoints(
    network: RoadNetwork,
    legs: Sequence[Sequence[int]],
    deliveries: Sequence[Sequence[tuple[int, int]]],
    depot: int,
) -> list[list[dict]]:
    """Per-vehicle ordered stops, each tied to a point index in its polyline.

    A page that shows which stop a vehicle is heading for needs three things the
    polyline alone cannot answer: the identity of each stop, the order, and where
    along the drawn path it falls.

    Two different indices are in play and conflating them silently mislabels every
    stop.  A delivery position from :func:`~qitransit.simulation.expand_solution`
    is a *position in that vehicle's path*, not an edge id, and the polyline adds
    the first edge's tail as an extra leading point, so a delivery at path
    position ``p`` lands on polyline point ``p + 1`` and the edge that ends it is
    ``path[p]``.  Reading ``p`` as an edge id still yields a plausible coordinate,
    which is why this reads as a working feature with the wrong stops labelled.
    """
    coords = network.coordinates()

    def point(node: int) -> list[float]:
        return [round(float(coords[node][0]), 1), round(float(coords[node][1]), 1)]

    depot_xy = point(depot)
    out: list[list[dict]] = []
    for index, edges in enumerate(legs):
        reached = deliveries[index] if index < len(deliveries) else ()
        marks: list[dict] = [{"at": 0, "kind": "depot", "label": "Depot", "xy": depot_xy}]
        for position, customer in reached:
            position = int(position)
            if not 0 <= position < len(edges):
                raise IndexError(
                    f"vehicle {index}: delivery for customer {customer} sits at path position "
                    f"{position}, outside its {len(edges)}-edge route"
                )
            marks.append(
                {
                    "at": position + 1,
                    "kind": "customer",
                    "label": f"C{int(customer) + 1}",
                    "xy": point(int(network.edges[edges[position]].to_node)),
                }
            )
        if edges:
            marks.append(
                {
                    "at": len(edges),
                    "kind": "depot",
                    "label": "Depot",
                    "xy": point(int(network.edges[edges[-1]].to_node)),
                }
            )
        out.append(marks)
    return out


def _failed(key: str, name: str, family: str, exc: Exception, wall: float, customers: int) -> MethodRun:
    return MethodRun(
        key=key,
        name=name,
        family=family,
        objective=float("inf"),
        feasible=False,
        vehicles=0,
        unserved=customers,
        penalty=float("inf"),
        eval_time=0.0,
        eval_distance=0.0,
        eval_externality=0.0,
        n_evals=0,
        wall_time=wall,
        error=f"{type(exc).__name__}: {exc}",
    )


def run_comparison(
    network: RoadNetwork,
    net_file: str | Path,
    problem: Problem,
    customer_nodes: Sequence[int],
    depot: int,
    edge_cost: np.ndarray,
    *,
    methods: Sequence[str] = DEFAULT_METHODS,
    budget: int = 3000,
    seed: int = 1,
    steps: int = 7200,
    sumo_seed: int = 42,
    simulate: bool = True,
    trace_every: int = 10,
    routes_dir: str | Path = "data/demo/routes",
    progress: Progress = _noop,
) -> Comparison:
    """Solve with every method, simulate every solution, return the comparison."""
    out_dir = Path(routes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = np.asarray(customer_nodes, dtype=np.int64)
    coords = network.coordinates()

    meta = {
        "network": Path(net_file).stem,
        "net_file": str(net_file),
        "instance": dict(problem.notes),
        "problem_family": problem.family,
        "customers": int(problem.n_customers),
        "budget": int(budget),
        "seed": int(seed),
        "sumo_seed": int(sumo_seed),
        "sumo_steps": int(steps),
        "depot": [round(float(coords[depot][0]), 1), round(float(coords[depot][1]), 1)],
        "points": [[round(float(x), 1), round(float(y), 1)] for x, y in coords[nodes]],
        "bounds": [
            round(float(coords[:, 0].min()), 1),
            round(float(coords[:, 1].min()), 1),
            round(float(coords[:, 0].max()), 1),
            round(float(coords[:, 1].max()), 1),
        ],
        "methods_available": sorted(REGISTRY),
    }
    comparison = Comparison(meta=meta, basemap=basemap(network))
    stages = 2 if simulate else 1
    total = max(len(methods) * stages, 1)

    for position, key in enumerate(methods):
        entry = get(key)
        progress(f"{entry.name}: optimising", position / total)
        started = time.perf_counter()
        try:
            result = entry.build(problem, budget=budget, seed=seed, sample_every=25).run()
        except Exception as exc:  # noqa: BLE001 - a failed method is a visible cell
            comparison.runs.append(
                _failed(key, entry.name, entry.family, exc, time.perf_counter() - started, problem.n_customers)
            )
            continue

        evaluation = result.evaluation
        arrays = result.convergence.as_arrays() if result.convergence else {}
        run = MethodRun(
            key=key,
            name=entry.name,
            family=entry.family,
            objective=float(evaluation.objective),
            feasible=bool(evaluation.feasible),
            vehicles=int(evaluation.vehicles),
            unserved=int(evaluation.unserved),
            penalty=float(evaluation.penalty),
            eval_time=float(evaluation.time),
            eval_distance=float(evaluation.distance),
            eval_externality=float(evaluation.externality),
            n_evals=int(result.n_evals),
            wall_time=float(result.wall_time),
            trace_evals=[int(v) for v in arrays.get("evals", [])],
            trace_costs=[float(v) for v in arrays.get("costs", [])],
        )

        legs, stops, _, deliveries = expand_solution(
            network, result.solution, depot, list(customer_nodes), edge_cost
        )
        run.polylines = polylines(network, legs)
        run.waypoints = waypoints(network, legs, deliveries, depot)
        run.routes_file = str(
            write_routes(
                out_dir / f"{key}.rou.xml",
                build_assignments(network, legs, stops, departs=departure_schedule(len(legs), seed=seed)),
            )
        )

        if simulate and legs:
            progress(f"{entry.name}: simulating in SUMO", (position + 0.5) / total)
            run.simulation = _simulate(net_file, run.routes_file, steps, sumo_seed, trace_every)
        comparison.runs.append(run)
        progress(f"{entry.name}: done", (position + 1) / total)

    return comparison


def _simulate(net_file: str | Path, routes_file: str, steps: int, seed: int, trace_every: int) -> dict:
    """Run one routes file and return its report, or a reason it could not run."""
    try:
        runner = SumoRunner(net_file, seed=seed)
        return runner.run(routes_file, steps=steps, sample_every=60, trace=True, trace_every=trace_every).as_dict()
    except Exception as exc:  # noqa: BLE001 - report the failure, keep the comparison
        return {"error": f"{type(exc).__name__}: {exc}"}
