"""Traffic state and edge cost models.

Route quality in this platform is not a single number.  The objective mixes
free-flow time, distance, and the congestion externality a route imposes on
everyone else in the network, and every component comes from the same
:class:`TrafficState` so the objective and the simulator stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..graph.network import RoadNetwork

BPR_ALPHA = 0.15
BPR_BETA = 4.0

CO2_G_PER_KM = 0.171
CO2_G_PER_KM_IDLE = 0.243


@dataclass(frozen=True, slots=True)
class TrafficState:
    """A snapshot of congestion, one value per network edge.

    ``speed_ratio`` is measured speed over free-flow speed, so ``1.0`` means
    free flow and ``0.0`` means standstill.  ``time_loss`` is the seconds of
    delay accumulated on the edge in the last step, which is what SUMO reports
    per edge and what the congestion externality is built from.
    """

    time: float
    speed_ratio: np.ndarray
    time_loss: np.ndarray
    vehicles: np.ndarray

    @classmethod
    def free_flow(cls, network: RoadNetwork, time: float = 0.0) -> TrafficState:
        zeros = np.zeros(network.n_edges, dtype=np.float64)
        return cls(time=time, speed_ratio=np.ones(network.n_edges), time_loss=zeros, vehicles=zeros)

    def bpr_delay(self, free_flow: np.ndarray, alpha: float = BPR_ALPHA, beta: float = BPR_BETA) -> np.ndarray:
        """BPR delay factor ``1 + alpha * (v/c)^beta - 1`` expressed as a multiplier."""
        congestion = np.clip(1.0 - self.speed_ratio, 0.0, 1.0) / 0.85
        return 1.0 + alpha * np.power(np.maximum(congestion, 0.0), beta)

    def travel_times(self, free_flow: np.ndarray) -> np.ndarray:
        """Per-edge traversal time in seconds implied by this state."""
        ratio = np.clip(self.speed_ratio, 0.05, 1.0)
        return free_flow / ratio


@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    """Weights of the multi-criteria routing objective.

    The default configuration minimises total travel time while charging for
    distance and for the delay a route pushes onto other road users, which is
    the standard behaviour of a system-optimal rather than a user-optimal
    assignment.
    """

    time: float = 1.0
    distance: float = 0.0
    congestion: float = 0.5
    vehicles: float = 0.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.time, self.distance, self.congestion, self.vehicles)


@runtime_checkable
class CostModel(Protocol):
    """Anything that can price a traversal of every edge."""

    name: str

    def edge_costs(self, weights: ObjectiveWeights) -> np.ndarray:
        """Per-edge cost vector, indexed like ``network.edges``."""


class FreeFlowCostModel:
    """Static free-flow travel time, optionally blended with distance."""

    def __init__(self, network: RoadNetwork) -> None:
        self.network = network
        self.free_flow = network.free_flow_time()
        self.name = "free-flow"

    def edge_costs(self, weights: ObjectiveWeights) -> np.ndarray:
        return weights.time * self.free_flow + weights.distance * self.network.lengths()


class StaticCongestionCostModel:
    """A fixed congestion snapshot, so benchmarks stay bit-for-bit repeatable.

    ``speed_ratio`` is either given directly or sampled once from a seeded
    generator and then frozen for the whole run.
    """

    def __init__(
        self,
        network: RoadNetwork,
        speed_ratio: np.ndarray | None = None,
        seed: int = 0,
        severity: float = 0.45,
    ) -> None:
        self.network = network
        self.free_flow = network.free_flow_time()
        if speed_ratio is None:
            rng = np.random.default_rng(seed)
            arterial = np.array(
                [1.0 - 0.55 * severity for e in network.edges if e.road_class == "arterial"]
                or [1.0]
            )
            ratio = rng.uniform(0.45, 1.0, size=network.n_edges) ** 1.5
            for i, edge in enumerate(network.edges):
                if edge.road_class == "arterial":
                    ratio[i] = min(ratio[i], 0.75)
                elif edge.road_class == "local":
                    ratio[i] = min(1.0, ratio[i] * 1.08)
            del arterial
            speed_ratio = ratio
        self.state = TrafficState(
            time=0.0,
            speed_ratio=np.clip(np.asarray(speed_ratio, dtype=np.float64), 0.05, 1.0),
            time_loss=np.zeros(network.n_edges),
            vehicles=np.zeros(network.n_edges),
        )
        self.name = "static-congestion"

    def edge_costs(self, weights: ObjectiveWeights) -> np.ndarray:
        times = self.state.travel_times(self.free_flow)
        delay = np.maximum(times - self.free_flow, 0.0)
        return weights.time * times + weights.distance * self.network.lengths() + weights.congestion * delay


class LiveCostModel:
    """Reads the live TraCI state and prices edges on the current snapshot.

    This is what makes the platform a *dynamic* router: the optimiser is handed
    a new cost vector every time step and re-plans against observed speeds.
    """

    def __init__(self, network: RoadNetwork, connection=None) -> None:
        self.network = network
        self.free_flow = network.free_flow_time()
        self.connection = connection
        self.name = "traci-live"
        self.state = TrafficState.free_flow(network)
        self._edge_ids = [e.id for e in network.edges]

    def refresh(self, time: float | None = None) -> np.ndarray:
        """Pull mean speeds for every edge from the running simulation."""
        if self.connection is None:
            return self.state.speed_ratio
        try:
            speeds = self.connection.edge.getLastStepMeanSpeed(self._edge_ids)
            vehicles = self.connection.edge.getLastStepVehicleNumber(self._edge_ids)
            losses = self.connection.edge.getLastStepTimeLoss(self._edge_ids)
        except Exception:
            return self.state.speed_ratio
        free = np.maximum(self.network.speeds(), 1e-3)
        ratio = np.asarray(speeds, dtype=np.float64) / free
        self.state = TrafficState(
            time=float(self.connection.simulation.getTime() if time is None else time),
            speed_ratio=np.clip(ratio, 0.05, 1.0),
            time_loss=np.asarray(losses, dtype=np.float64),
            vehicles=np.asarray(vehicles, dtype=np.float64),
        )
        return self.state.speed_ratio

    def edge_costs(self, weights: ObjectiveWeights) -> np.ndarray:
        times = self.state.travel_times(self.free_flow)
        externality = np.maximum(times - self.free_flow, 0.0) * np.clip(
            self.state.vehicles, 0.0, None
        ) / max(np.max(self.state.vehicles, initial=1.0), 1.0)
        return (
            weights.time * times
            + weights.distance * self.network.lengths()
            + weights.congestion * externality
        )


def snapshot_from_simulation(
    network: RoadNetwork,
    connection,
    time: float | None = None,
) -> TrafficState:
    """Read a full :class:`TrafficState` out of a live SUMO connection."""
    ids = [e.id for e in network.edges]
    speeds = np.asarray(connection.edge.getLastStepMeanSpeed(ids), dtype=np.float64)
    vehicles = np.asarray(connection.edge.getLastStepVehicleNumber(ids), dtype=np.float64)
    losses = np.asarray(connection.edge.getLastStepTimeLoss(ids), dtype=np.float64)
    free = np.maximum(network.speeds(), 1e-3)
    return TrafficState(
        time=float(connection.simulation.getTime() if time is None else time),
        speed_ratio=np.clip(speeds / free, 0.05, 1.0),
        time_loss=losses,
        vehicles=vehicles,
    )


def congestion_index(state: TrafficState) -> float:
    """Mean normalised delay, a scalar reading of how jammed the network is."""
    return float(np.mean(1.0 - state.speed_ratio))
