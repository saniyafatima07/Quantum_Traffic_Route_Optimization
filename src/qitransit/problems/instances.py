"""Instance generation: turning a road network into benchmark problems.

Every instance is a pure function of ``(network, seed, size)``, so a result in
the report can be reproduced by re-running one command.  Customers are placed
by sampling real junctions in the network rather than by inventing coordinates,
which is what makes the leg times — and therefore the congestion pressure —
representative of the city being studied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..graph.network import RoadNetwork
from ..routing.traffic import ObjectiveWeights, StaticCongestionCostModel
from .base import Customer
from .fleet import FleetVRP, Vehicle
from .graph_routing import CongestedShortestPath, TravelingSalesman
from .travel import TravelMatrix, build_travel_matrix


@dataclass(frozen=True, slots=True)
class InstanceSettings:
    """Everything that distinguishes one benchmark instance from another.

    Capacity is *derived* rather than set directly: it is sized so that the
    demand the generator produces fills ``target_vehicles`` vehicles.  Setting a
    capacity up front instead would either make the fleet trivially fit
    everything in one vehicle or strand customers whose demand exceeds it.
    """

    name: str
    network: str
    customers: int
    seed: int
    target_vehicles: int = 5
    service_time: float = 180.0
    horizon: float = 28800.0
    window_width_fraction: float = 0.35
    use_time_windows: bool = False
    congestion_severity: float = 0.5
    demand_low: int = 4
    demand_high: int = 21

    @property
    def capacity(self) -> float:
        mean_demand = 0.5 * (self.demand_low + self.demand_high)
        return max(
            1.0, mean_demand * max(self.customers - 1, 1) / max(self.target_vehicles, 1)
        )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "network": self.network,
            "customers": self.customers,
            "seed": self.seed,
            "target_vehicles": self.target_vehicles,
            "capacity": round(self.capacity, 2),
            "horizon_s": self.horizon,
            "time_windows": self.use_time_windows,
        }


def congestion_snapshot(
    network: RoadNetwork, seed: int, severity: float = 0.5
) -> tuple[np.ndarray, ObjectiveWeights]:
    """A fixed congestion state and the objective weights priced against it.

    Freezing the snapshot is deliberate: every algorithm sees identical travel
    times, so any difference in the result is attributable to the search and
    not to a different rush hour.
    """
    model = StaticCongestionCostModel(network, seed=seed, severity=severity)
    weights = ObjectiveWeights(time=1.0, distance=0.0, congestion=0.5)
    return model.edge_costs(weights), weights


def sample_nodes(
    network: RoadNetwork, count: int, seed: int, min_degree: int = 2
) -> np.ndarray:
    """Distinct junctions with at least ``min_degree`` outgoing edges.

    Dead ends and stubs are excluded because a customer placed there is
    unreachable in a useful direction and would only add noise to the leg
    matrix.
    """
    rng = np.random.default_rng(seed)
    degree = np.diff(network._out_start)
    eligible = np.flatnonzero(degree >= min_degree)
    if eligible.size < count:
        eligible = np.arange(network.n_nodes)
    return rng.choice(eligible, size=min(count, eligible.size), replace=False)


def sample_near(
    network: RoadNetwork,
    anchors: list[int],
    count: int,
    seed: int,
    radius_m: float = 2500.0,
    min_degree: int = 2,
) -> np.ndarray:
    """Junctions near a set of anchor nodes, for clustered urban demand.

    Real delivery demand is not uniform: it concentrates around commercial
    corridors.  Sampling near named landmarks reproduces that clustering, which
    in turn makes the congestion term of the objective do real work.
    """
    rng = np.random.default_rng(seed)
    coords = network.coordinates()
    degree = np.diff(network._out_start)
    chosen: list[int] = []
    for anchor in anchors:
        if len(chosen) >= count:
            break
        distances = np.hypot(
            coords[:, 0] - coords[anchor, 0], coords[:, 1] - coords[anchor, 1]
        )
        pool = np.flatnonzero((distances <= radius_m) & (degree >= min_degree))
        if pool.size == 0:
            pool = np.flatnonzero(degree >= min_degree)
        take = max(1, count // max(len(anchors), 1))
        chosen.extend(int(n) for n in rng.choice(pool, size=min(take, pool.size), replace=False))
    unique = list(dict.fromkeys(chosen))
    if len(unique) < count:
        seen = set(unique)
        for node in sample_nodes(network, count * 2, seed, min_degree):
            if int(node) not in seen:
                unique.append(int(node))
                seen.add(int(node))
            if len(unique) >= count:
                break
    return np.array(unique[:count], dtype=np.int64)


def make_customers(
    nodes: np.ndarray,
    settings: InstanceSettings,
    rng: np.random.Generator,
    depot_index: int = 0,
) -> list[Customer]:
    """Demands, service times and time windows for a set of customer junctions."""
    customers: list[Customer] = []
    horizon = settings.horizon
    for position, node in enumerate(nodes):
        if position == depot_index:
            continue
        demand = float(rng.integers(settings.demand_low, settings.demand_high))
        window_width = horizon * settings.window_width_fraction * float(rng.uniform(0.7, 1.3))
        latest = horizon
        earliest = max(0.0, latest - window_width)
        centre = earliest + (latest - earliest) * float(rng.uniform(0.15, 0.85))
        half = window_width / 2.0
        customers.append(
            Customer(
                node=int(node),
                demand=demand,
                service_time=settings.service_time,
                window_start=max(0.0, centre - half),
                window_end=min(horizon, centre + half),
                name=f"c{position - 1:03d}",
            )
        )
    return customers


def make_fleet_instance(
    network: RoadNetwork,
    nodes: np.ndarray,
    settings: InstanceSettings,
    edge_cost: np.ndarray,
    weights: ObjectiveWeights,
    matrix: TravelMatrix | None = None,
) -> FleetVRP:
    """A single-depot capacitated (optionally time-windowed) instance."""
    rng = np.random.default_rng(settings.seed + 7)
    if matrix is None:
        matrix = build_travel_matrix(network, nodes, edge_cost, weights)
    customers = make_customers(nodes, settings, rng)
    capacity = _capacity_for(customers, settings)
    return FleetVRP(
        network,
        matrix,
        int(nodes[0]),
        customers,
        [Vehicle(capacity)],
        name=settings.name,
        seed=settings.seed,
        use_time_windows=settings.use_time_windows,
        weights=weights.as_tuple(),
        network_name=settings.network,
        notes=settings.as_dict(),
    )


def _capacity_for(customers: list[Customer], settings: InstanceSettings) -> float:
    """The capacity that makes the generated demand fill the target fleet.

    A five per cent margin above the exact split keeps the instance feasible
    while still leaving the number of vehicles genuinely contested: a solver has
    to choose *how* to pack, not whether the load fits at all.
    """
    total = sum(customer.demand for customer in customers)
    per_vehicle = total / max(settings.target_vehicles, 1)
    return max(per_vehicle * 1.05, max((c.demand for c in customers), default=1.0))


def make_tsp_instance(
    network: RoadNetwork,
    nodes: np.ndarray,
    settings: InstanceSettings,
    edge_cost: np.ndarray,
    weights: ObjectiveWeights,
    matrix: TravelMatrix | None = None,
) -> TravelingSalesman:
    """A single-vehicle tour over the same junctions."""
    if matrix is None:
        matrix = build_travel_matrix(network, nodes, edge_cost, weights)
    return TravelingSalesman(
        matrix,
        name=settings.name,
        seed=settings.seed,
        weights=weights.as_tuple(),
        network_name=settings.network,
        notes=settings.as_dict(),
    )


def make_spp_instance(
    network: RoadNetwork,
    hub: int,
    targets: np.ndarray,
    settings: InstanceSettings,
    edge_cost: np.ndarray,
    n_vehicles: int = 12,
) -> CongestedShortestPath:
    """A hub dispatching vehicles under a shared congestion charge."""
    return CongestedShortestPath(
        network,
        int(hub),
        [int(t) for t in targets],
        edge_cost,
        n_vehicles=n_vehicles,
        congestion_strength=0.35,
        name=settings.name,
        seed=settings.seed,
        network_name=settings.network,
        notes=settings.as_dict(),
    )


def instance_settings(
    network_name: str,
    customers: int,
    seed: int,
    *,
    use_time_windows: bool = False,
    horizon: float = 28800.0,
    congestion_severity: float = 0.5,
    target_vehicles: int = 5,
) -> InstanceSettings:
    return InstanceSettings(
        name=f"{network_name}-n{customers}-s{seed}{'-tw' if use_time_windows else ''}",
        network=network_name,
        customers=customers,
        seed=seed,
        horizon=horizon,
        use_time_windows=use_time_windows,
        congestion_severity=congestion_severity,
        target_vehicles=target_vehicles,
    )
