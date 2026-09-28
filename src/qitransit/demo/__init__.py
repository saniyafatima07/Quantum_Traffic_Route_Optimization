"""The SUMO comparison demo.

A local web page that runs every registered method on one real Bengaluru
instance, drives each method's routes through SUMO, and shows the two side by
side.  Nothing here invents its own measurement: the optimisers come from
:mod:`qitransit.algorithms`, the instance from :mod:`qitransit.problems`, and the
simulated numbers from :mod:`qitransit.simulation`, so what the page shows is
what the rest of the platform produces.
"""

from __future__ import annotations

from .pipeline import (
    DEFAULT_METHODS,
    Comparison,
    MethodRun,
    basemap,
    polylines,
    run_comparison,
    waypoints,
)
from .server import Builder, DemoService, serve

__all__ = [
    "DEFAULT_METHODS",
    "Builder",
    "Comparison",
    "DemoService",
    "MethodRun",
    "basemap",
    "polylines",
    "run_comparison",
    "serve",
    "waypoints",
]
