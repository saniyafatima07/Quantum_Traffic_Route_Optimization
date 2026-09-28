"""The benchmark protocol.

Two rules make the comparison defensible, and everything here exists to enforce
them:

**Equal evaluation budget.**  Every method gets the same number of *objective
evaluations*, not the same number of iterations or the same wall-clock time.
Iterations are meaningless across a swarm, a genetic algorithm and a local
search — one GA iteration can cost a population's worth of evaluations — and
wall clock punishes the pure-Python decoders differently from a compiled
construction heuristic.  Counting evaluations is the only common currency that
counts the same work for everyone.  Wall time is still recorded, and reported,
because it is what a user actually waits.

**Identical problem instances.**  Every method sees the same decoders, the same
objective weights and the same frozen congestion snapshot, so a difference in the
result is attributable to the search and nothing else.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..algorithms import Entry, get
from ..algorithms.base import OptimizerResult
from ..problems.base import Problem
from .records import RunRecord, target_per_instance


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    """One benchmark sweep: which methods, which instances, how much budget."""

    name: str
    keys: tuple[str, ...]
    seeds: tuple[int, ...]
    budget: int
    sample_every: int = 25
    problem_family: str = "cvrp"

    def describe(self) -> str:
        return (
            f"{self.name}: {len(self.keys)} methods x {len(self.seeds)} seeds "
            f"x {self.budget} evaluations"
        )


def run_suite(
    spec: SuiteSpec,
    problems: Sequence[tuple[str, Problem]],
    *,
    progress: bool = True,
) -> list[RunRecord]:
    """Run every method on every instance for every seed, and record the lot.

    A method that raises is recorded as infeasible with an infinite objective
    rather than aborting the sweep: losing the whole run to one bad
    configuration would make the remaining methods unrunnable, and a missing
    cell in a results table is a visible failure while a silent one is not.
    """
    raw: list[tuple[str, Entry, OptimizerResult]] = []
    for instance_name, problem in problems:
        for key in spec.keys:
            entry = get(key)
            for seed in spec.seeds:
                started = time.perf_counter()
                try:
                    result = entry.build(
                        problem, budget=spec.budget, seed=seed, sample_every=spec.sample_every
                    ).run()
                except Exception as exc:  # noqa: BLE001 - a failed cell is data
                    if progress:
                        print(f"    {entry.name:10s} {instance_name} seed={seed} FAILED: {exc}")
                    raw.append((instance_name, entry, _failed_result(problem, entry, seed, exc)))
                    continue
                if progress:
                    print(
                        f"    {entry.name:10s} {instance_name} seed={seed} "
                        f"obj={result.evaluation.objective:12.2f} "
                        f"evals={result.n_evals} wall={time.perf_counter() - started:6.2f}s",
                        flush=True,
                    )
                raw.append((instance_name, entry, result))

    records = [_to_record(spec, name, entry, result) for name, entry, result in raw]
    targets = target_per_instance(records)
    for record in records:
        target = targets.get(record.instance_key)
        if target is not None and np.isfinite(target):
            record.evals_to_target = record.evals_to_best and None
            record.evals_to_target = record.convergence_evals_to(target)
    return records


def _failed_result(problem: Problem, entry: Entry, seed: int, exc: Exception) -> OptimizerResult:
    from ..problems.base import Evaluation

    evaluation = Evaluation(
        objective=float("inf"),
        time=0.0,
        distance=0.0,
        externality=0.0,
        vehicles=0,
        penalty=float("inf"),
        unserved=problem.n_customers,
        feasible=False,
    )
    return OptimizerResult(
        algorithm=entry.name,
        problem=problem.name,
        seed=seed,
        solution=(),
        evaluation=evaluation,
        convergence=None,  # type: ignore[arg-type]
        n_evals=0,
        wall_time=0.0,
        meta={"error": f"{type(exc).__name__}: {exc}"},
    )


def _to_record(
    spec: SuiteSpec, instance_name: str, entry: Entry, result: OptimizerResult
) -> RunRecord:
    convergence = result.convergence
    arrays = convergence.as_arrays() if convergence is not None else {}
    evaluation = result.evaluation
    evals_to_best = None
    if arrays:
        final = float(arrays["costs"][-1])
        evals_to_best = int(arrays["evals"][-1]) if arrays["costs"][-1] == final else None
    return RunRecord(
        suite=spec.name,
        method=entry.name,
        family=entry.family,
        instance=instance_name,
        problem_family=spec.problem_family,
        seed=result.seed,
        budget=spec.budget,
        n_customers=result.n_evals and int(result.evaluation.unserved) or 0,
        network=str(result.meta.get("network", "")),
        objective=float(evaluation.objective),
        feasible=bool(evaluation.feasible),
        eval_time=float(evaluation.time),
        eval_distance=float(evaluation.distance),
        eval_externality=float(evaluation.externality),
        vehicles=int(evaluation.vehicles),
        penalty=float(evaluation.penalty),
        unserved=int(evaluation.unserved),
        n_evals=int(result.n_evals),
        wall_time=float(result.wall_time),
        lower_bound=float(result.meta.get("lower_bound", float("inf"))),
        evals_to_best=evals_to_best,
        solution=[list(route) for route in result.solution],
        trace_evals=arrays.get("evals", np.empty(0, dtype=np.int64)).tolist(),
        trace_costs=arrays.get("costs", np.empty(0)).tolist(),
        trace_times=arrays.get("times", np.empty(0)).tolist(),
    )


def summarise(records: Iterable[RunRecord]) -> list[dict]:
    """Per-method aggregate: mean objective, spread, feasibility, time."""
    grouped: dict[str, list[RunRecord]] = {}
    for record in records:
        grouped.setdefault(record.method, []).append(record)
    rows: list[dict] = []
    for method, group in grouped.items():
        objectives = np.array([r.objective for r in group], dtype=np.float64)
        feasible = np.array([r.feasible for r in group], dtype=bool)
        finite = objectives[np.isfinite(objectives)]
        rows.append(
            {
                "method": method,
                "family": group[0].family,
                "runs": len(group),
                "feasible": int(feasible.sum()),
                "mean_objective": float(finite.mean()) if finite.size else float("nan"),
                "median_objective": float(np.median(finite)) if finite.size else float("nan"),
                "worst_objective": float(finite.max()) if finite.size else float("nan"),
                "mean_wall_time": float(np.mean([r.wall_time for r in group])),
                "total_wall_time": float(np.sum([r.wall_time for r in group])),
                "mean_vehicles": float(np.mean([r.vehicles for r in group])),
                "mean_time_s": float(np.mean([r.eval_time for r in group])),
                "mean_distance_m": float(np.mean([r.eval_distance for r in group])),
            }
        )
    rows.sort(key=lambda row: (np.isnan(row["mean_objective"]), row["mean_objective"]))
    return rows
