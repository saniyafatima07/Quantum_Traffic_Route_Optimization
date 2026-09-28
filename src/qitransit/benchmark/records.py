"""Result records: the on-disk form of everything a benchmark run produced.

Records are plain JSON so that a report can be regenerated months later without
re-running the search, and so that a partially finished sweep is still readable.
The convergence trace is stored alongside the scalar summary because the claim
the project makes is about *convergence speed*, not only final quality.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


@dataclass(slots=True)
class RunRecord:
    """One (method, instance, seed) run."""

    suite: str
    method: str
    family: str
    instance: str
    problem_family: str
    seed: int
    budget: int
    n_customers: int
    network: str
    objective: float
    feasible: bool
    eval_time: float
    eval_distance: float
    eval_externality: float
    vehicles: int
    penalty: float
    unserved: int
    n_evals: int
    wall_time: float
    lower_bound: float = float("inf")
    evals_to_best: int | None = None
    evals_to_target: int | None = None
    solution: list[list[int]] = field(default_factory=list)
    trace_evals: list[int] = field(default_factory=list)
    trace_costs: list[float] = field(default_factory=list)
    trace_times: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> RunRecord:
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in fields})

    @property
    def instance_key(self) -> str:
        return f"{self.problem_family}/{self.instance}"

    def gap_against(self, best: float) -> float:
        """Relative excess over ``best``; zero when the run matched it."""
        if not np.isfinite(best) or best == 0.0:
            return 0.0
        return (self.objective - best) / abs(best)


def records_dir(root: str | Path = "results") -> Path:
    return Path(root) / "raw"


def write_records(records: Iterable[RunRecord], root: str | Path = "results") -> Path:
    """Write one JSON-lines file per suite, and return the directory."""
    target = records_dir(root)
    target.mkdir(parents=True, exist_ok=True)
    by_suite: dict[str, list[RunRecord]] = {}
    for record in records:
        by_suite.setdefault(record.suite, []).append(record)
    for suite, group in by_suite.items():
        path = target / f"{suite}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in group:
                handle.write(json.dumps(record.as_dict()) + "\n")
    return target


def read_records(suite: str, root: str | Path = "results") -> list[RunRecord]:
    path = records_dir(root) / f"{suite}.jsonl"
    if not path.is_file():
        return []
    out: list[RunRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(RunRecord.from_dict(json.loads(line)))
    return out


def iter_suites(root: str | Path = "results") -> Iterator[str]:
    target = records_dir(root)
    if not target.is_dir():
        return
    for path in sorted(target.glob("*.jsonl")):
        yield path.stem


def best_per_instance(records: Iterable[RunRecord]) -> dict[str, float]:
    """The best objective any method reached on each instance.

    Instances are keyed by ``(problem_family, instance)`` rather than by name
    alone, so a suite that reuses a name across families cannot silently pool
    incomparable objectives.
    """
    best: dict[str, float] = {}
    for record in records:
        key = record.instance_key
        value = float(record.objective)
        if np.isfinite(value) and (key not in best or value < best[key]):
            best[key] = value
    return best


def target_per_instance(
    records: Iterable[RunRecord], fraction: float = 1.02
) -> dict[str, float]:
    """A common target per instance: ``fraction`` times its best objective.

    Convergence speed is only comparable across methods when they are racing
    toward the same mark, and the mark has to come from outside any single
    method — hence the best over the whole sweep rather than the best of the
    method being timed.
    """
    return {key: value * fraction for key, value in best_per_instance(records).items()}
