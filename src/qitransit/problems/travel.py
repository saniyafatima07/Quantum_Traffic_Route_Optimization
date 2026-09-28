"""Pre-computed travel information between the points a route must visit.

Repeated route construction inside an optimiser's inner loop cannot afford a
graph search per leg, so every instance builds a small table once: the points
are the depot plus the customers, and each cell holds the objective cost, the
travel time, the distance and the congestion externality of the best leg
between two points.

The leg is the one that minimises the *weighted* objective rather than pure
time, which is what makes a congestion-aware solution genuinely different from
a shortest-in-time one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..graph.network import RoadNetwork
from ..routing.shortest_path import dijkstra_tree
from ..routing.traffic import ObjectiveWeights


@dataclass(frozen=True, slots=True)
class TravelMatrix:
    """Leg costs between depot and customers.

    Attributes
    ----------
    points:
        Node indices, ``points[0]`` being the depot.
    cost / time / distance / externality:
        ``[n, n]`` arrays; ``[i, j]`` describes travelling from ``i`` to ``j``.
    """

    points: np.ndarray
    cost: np.ndarray
    time: np.ndarray
    distance: np.ndarray
    externality: np.ndarray
    free_flow: np.ndarray
    legs: dict[tuple[int, int], tuple[int, ...]] | None = None

    def __len__(self) -> int:
        return int(self.points.size)

    def leg(self, i: int, j: int) -> tuple[int, ...]:
        """Edge sequence realising the leg, computed on demand."""
        if self.legs is not None:
            return self.legs.get((i, j), ())
        return ()

    def depot_time(self) -> np.ndarray:
        return self.time[0]


def build_travel_matrix(
    network: RoadNetwork,
    points: np.ndarray,
    edge_cost: np.ndarray,
    weights: ObjectiveWeights,
    keep_legs: bool = False,
    progress: bool = False,
) -> TravelMatrix:
    """One Dijkstra tree per point, then walk it once per destination.

    ``edge_cost`` is the already-weighted per-edge cost, so the tree it produces
    minimises the routing objective.  Free-flow time is carried alongside so
    that the externality of each leg can be reported separately.
    """
    n = int(points.size)
    lengths = network.lengths()
    free_flow = network.free_flow_time()
    point_of_node = np.full(network.n_nodes, -1, dtype=np.int64)
    point_of_node[points] = np.arange(n, dtype=np.int64)

    cost = np.zeros((n, n), dtype=np.float64)
    time = np.zeros((n, n), dtype=np.float64)
    distance = np.zeros((n, n), dtype=np.float64)
    externality = np.zeros((n, n), dtype=np.float64)
    ff_time = np.zeros((n, n), dtype=np.float64)
    legs: dict[tuple[int, int], tuple[int, ...]] | None = {} if keep_legs else None

    for src in range(n):
        source = int(points[src])
        dist, prev_node, prev_edge = dijkstra_tree(network, edge_cost, source)
        for dst_node in np.flatnonzero(point_of_node >= 0):
            if int(dst_node) == source:
                continue
            dst = int(point_of_node[dst_node])
            if not np.isfinite(dist[dst_node]):
                continue
            node = int(dst_node)
            leg: list[int] = []
            leg_time = 0.0
            leg_len = 0.0
            leg_ff = 0.0
            guard = 0
            while node != source and prev_edge[node] >= 0 and guard <= network.n_nodes:
                e = int(prev_edge[node])
                leg.append(e)
                leg_time += float(edge_cost[e])
                leg_len += float(lengths[e])
                leg_ff += float(free_flow[e])
                node = int(prev_node[node])
                guard += 1
            if not leg:
                continue
            leg.reverse()
            cost[src, dst] = float(dist[dst_node])
            time[src, dst] = leg_time
            distance[src, dst] = leg_len
            ff_time[src, dst] = leg_ff
            externality[src, dst] = leg_time - leg_ff
            if legs is not None:
                legs[(src, dst)] = tuple(leg)
        if progress and (src + 1) % 10 == 0:
            print(f"    travel matrix {src + 1}/{n}", flush=True)

    return TravelMatrix(
        points=points,
        cost=cost,
        time=time,
        distance=distance,
        externality=externality,
        free_flow=ff_time,
        legs=legs,
    )


def expand_route_legs(
    network: RoadNetwork,
    matrix: TravelMatrix,
    point_indices: Sequence[int],
) -> tuple[int, ...]:
    """Concatenate the leg edges of a sequence of matrix points into one edge path."""
    out: list[int] = []
    for a, b in zip(point_indices, point_indices[1:], strict=False):
        out.extend(matrix.leg(a, b))
    return tuple(out)


def nearest_point_leg(
    network: RoadNetwork, edge_cost: np.ndarray, tail: int, head: int
) -> tuple[int, ...]:
    """A one-off leg search for pairs outside a precomputed matrix."""
    from ..routing.shortest_path import dijkstra

    return dijkstra(network, edge_cost, tail, head).edges


def matrix_from_pairs(
    network: RoadNetwork,
    sources: np.ndarray,
    targets: np.ndarray,
    edge_cost: np.ndarray,
    weights: ObjectiveWeights,
) -> TravelMatrix:
    """Convenience wrapper for problems that need only a few ordered pairs."""
    points = np.unique(np.concatenate([sources, targets]))
    return build_travel_matrix(network, points, edge_cost, weights)


def mean_leg_time(matrix: TravelMatrix) -> float:
    """Average leg time, a sanity figure for instance generation."""
    n = len(matrix)
    return float(matrix.time.sum() / max(n * (n - 1), 1))


def objective_weights_from_dict(payload: dict) -> ObjectiveWeights:
    return ObjectiveWeights(
        time=float(payload.get("time", 1.0)),
        distance=float(payload.get("distance", 0.0)),
        congestion=float(payload.get("congestion", 0.5)),
        vehicles=float(payload.get("vehicles", 0.0)),
    )
