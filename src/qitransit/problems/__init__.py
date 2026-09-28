"""Problem definitions, instance generation and their shared vocabulary."""

from __future__ import annotations

from .base import Customer, Evaluation, Problem, ProblemSpec, Route, Solution
from .fleet import FleetVRP, Vehicle, clarke_wright, sequential_savings
from .graph_routing import CongestedShortestPath, TravelingSalesman
from .instances import (
    InstanceSettings,
    congestion_snapshot,
    instance_settings,
    make_customers,
    make_fleet_instance,
    make_spp_instance,
    make_tsp_instance,
    sample_near,
    sample_nodes,
)
from .travel import TravelMatrix, build_travel_matrix

__all__ = [
    "CongestedShortestPath",
    "Customer",
    "Evaluation",
    "FleetVRP",
    "InstanceSettings",
    "Problem",
    "ProblemSpec",
    "Route",
    "Solution",
    "TravelMatrix",
    "TravelingSalesman",
    "Vehicle",
    "build_travel_matrix",
    "clarke_wright",
    "congestion_snapshot",
    "instance_settings",
    "make_customers",
    "make_fleet_instance",
    "make_spp_instance",
    "make_tsp_instance",
    "sample_near",
    "sample_nodes",
    "sequential_savings",
]
