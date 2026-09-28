"""Figures for the report.

Everything is written from stored records rather than from a live sweep, so a
figure can be regenerated without re-running a search that took hours.  The
convergence plots normalise each curve to its own instance before averaging:
absolute objectives differ by orders of magnitude between a 20-customer and an
80-customer instance, and averaging raw values would let the largest instance
alone determine the shape of every curve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .records import RunRecord, best_per_instance

QUANTUM = ("QPSO", "QPSO+LS", "QGA", "QISEP", "QISEP+LS")
CLASSICAL = ("PSO", "GA", "SA", "Savings")


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _colour(method: str) -> str:
    if method in ("QPSO", "QPSO+LS"):
        return "#1f6fb4"
    if method.startswith("QISEP"):
        return "#3aa6a0"
    if method == "QGA":
        return "#7b5ea7"
    if method == "PSO":
        return "#c1666b"
    if method == "GA":
        return "#d99a2b"
    if method == "SA":
        return "#8d8d8d"
    return "#4c4c4c"


def _linestyle(method: str) -> str:
    return "-" if method.startswith(("QPSO", "QISEP")) else "--"


def convergence_figure(
    records: Sequence[RunRecord], path: Path, title: str = ""
) -> Path:
    """Best-so-far against evaluations spent, per instance size, averaged."""
    plt = _matplotlib()
    sizes = sorted({r.n_customers for r in records})
    methods = sorted({r.method for r in records}, key=lambda m: (_rank(m), m))
    fig, axes = plt.subplots(1, max(len(sizes), 1), figsize=(4.2 * len(sizes), 3.8), squeeze=False)

    for column, size in enumerate(sizes):
        ax = axes[0][column]
        subset = [r for r in records if r.n_customers == size]
        best = best_per_instance(subset)
        for method in methods:
            curves = []
            for record in subset:
                if record.method != method or not record.trace_costs:
                    continue
                target = best.get(record.instance_key)
                if not target or not np.isfinite(target) or target == 0.0:
                    continue
                curves.append((np.asarray(record.trace_evals), np.asarray(record.trace_costs) / target))
            if not curves:
                continue
            grid = _common_grid(curves)
            mean = np.array([c[1][np.searchsorted(c[0], grid, side="right") - 1] for c in curves])
            ax.plot(grid, mean.mean(axis=0), _linestyle(method), color=_colour(method),
                    label=f"{method} (n={len(curves)})", linewidth=1.6)
        ax.set_title(f"{size} customers")
        ax.set_xlabel("objective evaluations")
        ax.set_ylabel("objective / best found")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _rank(method: str) -> int:
    order = ("QPSO", "QPSO+LS", "QISEP", "QISEP+LS", "QGA", "PSO", "GA", "SA", "Savings")
    return order.index(method) if method in order else len(order)


def _common_grid(curves: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    shortest = min(c[0][-1] for c in curves if c[0].size)
    grid = np.unique(np.concatenate([c[0][c[0] <= shortest] for c in curves if c[0].size]))
    return grid if grid.size <= 2000 else np.linspace(grid[0], grid[-1], 400)


def gap_figure(records: Sequence[RunRecord], path: Path) -> Path:
    """Where each method finishes relative to the best objective found."""
    plt = _matplotlib()
    best = best_per_instance(records)
    methods = sorted({r.method for r in records}, key=lambda m: (_rank(m), m))
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    data, labels = [], []
    for method in methods:
        gaps = [
            r.gap_against(best[r.instance_key])
            for r in records
            if r.method == method and r.instance_key in best and np.isfinite(r.objective)
        ]
        if gaps:
            data.append(np.array(gaps))
            labels.append(method)
    box = ax.boxplot(data, tick_labels=labels, showfliers=False)
    for patch, method in zip(box["boxes"], labels, strict=True):
        patch.set_facecolor(_colour(method))
        patch.set_alpha(0.55)
    ax.set_ylabel("gap to best found (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Solution quality across the sweep")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def quality_vs_time_figure(records: Sequence[RunRecord], path: Path) -> Path:
    """Objective against wall time: what a user actually waits for."""
    plt = _matplotlib()
    best = best_per_instance(records)
    methods = sorted({r.method for r in records}, key=lambda m: (_rank(m), m))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for method in methods:
        xs, ys = [], []
        for record in records:
            target = best.get(record.instance_key)
            if record.method != method or not target or not np.isfinite(record.objective):
                continue
            xs.append(max(record.wall_time, 1e-3))
            ys.append(record.objective / target)
        if xs:
            ax.scatter(xs, ys, s=14, alpha=0.55, color=_colour(method),
                       label=method, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall time per run (s)")
    ax.set_ylabel("objective / best found")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def all_figures(records: Iterable[RunRecord], out_dir: Path) -> list[Path]:
    records = list(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        return []
    return [
        convergence_figure(records, out_dir / "convergence.png"),
        gap_figure(records, out_dir / "gap.png"),
        quality_vs_time_figure(records, out_dir / "quality_vs_time.png"),
    ]
