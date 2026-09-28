"""Weighted directed graph model of a road network.

The model mirrors SUMO's own abstractions: a *node* is a junction, an *edge* is
a directed road between two junctions, and an edge carries a free-flow speed, a
length, a lane count and a priority.  Adjacency is stored in compressed form so
that the thousands of shortest-path and route evaluations performed by the
optimisers stay cheap.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class Node:
    """A junction in the network.

    ``x``/``y`` are in the network's own projected frame, which is what SUMO
    simulates with.  ``lat``/``lon`` carry the WGS84 position when the network
    came from a geographic source, and are ``None`` for purely synthetic
    layouts.
    """

    id: str
    index: int
    x: float
    y: float
    has_traffic_light: bool = False
    lat: float | None = None
    lon: float | None = None

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class Edge:
    """A directed road segment."""

    id: str
    index: int
    from_node: int
    to_node: int
    length: float
    speed: float
    lanes: int
    priority: int
    road_class: str

    @property
    def free_flow_time(self) -> float:
        """Travel time in seconds at the speed limit with no traffic."""
        return self.length / max(self.speed, 1e-6)


class RoadNetwork:
    """An immutable weighted digraph over junctions.

    Parameters
    ----------
    nodes:
        Junction records.  ``index`` must equal the position in the sequence.
    edges:
        Directed road records.  ``from_node``/``to_node`` are indices into
        ``nodes``.
    connections:
        The *turn graph*: for each edge, the edges a vehicle may legally take
        next.  Omit it and the relation is inferred from node adjacency, which
        permits every turn including U-turns.

    The turn graph is what separates a real network model from a graph
    abstraction.  A compiled SUMO network forbids U-turns at most junctions and
    applies per-lane access restrictions, so a router that knows only node
    adjacency will emit routes the simulator then refuses to load.  Storing
    connections here means the routing layer and the simulator read the same
    legality rules, and the routes an optimiser prices are routes that can
    actually be driven.
    """

    __slots__ = (
        "nodes",
        "edges",
        "n_nodes",
        "n_edges",
        "_out",
        "_in",
        "_out_start",
        "_out_target",
        "_in_start",
        "_in_source",
        "_lookup",
        "_free_flow",
        "_length",
        "_speed",
        "_tail",
        "_head",
        "_index_of_node_id",
        "_conn_start",
        "_conn_edges",
        "_conn_from",
        "_has_turn_graph",
        "_turn_cache",
        "_reverse_cache",
        "_coord_cache",
    )

    def __init__(
        self,
        nodes: Sequence[Node],
        edges: Sequence[Edge],
        connections: Sequence[Sequence[int]] | None = None,
    ) -> None:
        self.nodes = tuple(nodes)
        self.edges = tuple(edges)
        self.n_nodes = len(self.nodes)
        self.n_edges = len(self.edges)
        self._turn_cache: tuple | None = None
        self._reverse_cache: tuple | None = None
        self._coord_cache: np.ndarray | None = None
        self._index_of_node_id = {n.id: n.index for n in self.nodes}

        self._tail = np.fromiter(
            (e.from_node for e in edges), dtype=np.int32, count=self.n_edges
        )
        self._head = np.fromiter(
            (e.to_node for e in edges), dtype=np.int32, count=self.n_edges
        )
        self._length = np.fromiter(
            (e.length for e in edges), dtype=np.float64, count=self.n_edges
        )
        self._speed = np.fromiter(
            (e.speed for e in edges), dtype=np.float64, count=self.n_edges
        )
        speed_safe = np.maximum(self._speed, 1e-6)
        self._free_flow = self._length / speed_safe

        self._out_start, self._out_target, self._out = self._build_adjacency(
            self.n_nodes, self._tail, self._head
        )
        self._in_start, self._in_source, self._in = self._build_adjacency(
            self.n_nodes, self._head, self._tail
        )
        self._lookup = {
            (int(s), int(d)): i
            for i, (s, d) in enumerate(zip(self._tail, self._head, strict=True))
        }
        self._build_turn_graph(connections)

    def _build_turn_graph(self, connections: Sequence[Sequence[int]] | None) -> None:
        """Store the legal successor edges of every edge, in CSR form."""
        if connections is None:
            self._has_turn_graph = False
            rows = [self.out_edges(int(head)) for head in self._head]
        else:
            self._has_turn_graph = True
            rows = [
                np.unique(np.fromiter((int(f) for f in succ), dtype=np.int32))
                if len(succ)
                else np.empty(0, dtype=np.int32)
                for succ in connections
            ]
        counts = np.fromiter(
            (row.size for row in rows), dtype=np.int64, count=self.n_edges
        )
        start = np.zeros(self.n_edges + 1, dtype=np.int64)
        np.cumsum(counts, out=start[1:])
        self._conn_start = start
        self._conn_edges = (
            np.concatenate(rows)
            if self.n_edges and int(counts.sum())
            else np.empty(0, dtype=np.int32)
        )
        self._conn_from = np.repeat(
            np.arange(self.n_edges, dtype=np.int32), counts.astype(np.int64)
        )

    @staticmethod
    def _build_adjacency(
        n_nodes: int, src: np.ndarray, dst: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Node-indexed CSR: ``start`` over ``src``, with ``dst`` and the edge ids."""
        counts = np.bincount(src, minlength=n_nodes)
        start = np.zeros(n_nodes + 1, dtype=np.int64)
        np.cumsum(counts, out=start[1:])
        order = np.argsort(src, kind="stable")
        return start, dst[order], order

    @property
    def has_turn_graph(self) -> bool:
        """Whether legal turns came from the source or were inferred from nodes."""
        return self._has_turn_graph

    def successor_edges(self, edge: int) -> np.ndarray:
        """Edge indices a vehicle may take after traversing ``edge``."""
        e = int(edge)
        return self._conn_edges[self._conn_start[e] : self._conn_start[e + 1]]

    def predecessor_edges(self, edge: int) -> np.ndarray:
        """Edge indices from which a vehicle may legally enter ``edge``."""
        start, edges = self.reverse_turn_lists()
        e = int(edge)
        return np.array(edges[start[e] : start[e + 1]], dtype=np.int32)

    def may_turn(self, first: int, second: int) -> bool:
        """Whether ``first -> second`` is a legal turn."""
        return bool(np.any(self.successor_edges(first) == int(second)))

    @property
    def n_connections(self) -> int:
        return int(self._conn_edges.size)

    def turn_lists(self) -> tuple[list[int], list[int], list[int], list[int]]:
        """The turn graph as Python lists, built once and cached.

        The search inner loop runs in CPython, where indexing a numpy array
        costs far more than indexing a list, and it indexes these structures
        once per settled state.  Converting once per network rather than once
        per query is what keeps repeated routing affordable.
        """
        cached = self._turn_cache
        if cached is None:
            cached = (
                self._conn_start.tolist(),
                self._conn_edges.tolist(),
                self._head.tolist(),
                self._tail.tolist(),
            )
            self._turn_cache = cached
        return cached

    def reverse_turn_lists(self) -> tuple[list[int], list[int]]:
        """The turn graph reversed: for each edge, the edges that may precede it.

        Grouped by the *successor* side of each connection, since a backward
        search from an edge must expand the edges allowed to feed it.  Grouping
        by the predecessor side would hand back the forward successors and the
        backward search would wander away from the target instead of closing in
        on it.
        """
        cached = self._reverse_cache
        if cached is None:
            counts = np.bincount(self._conn_edges, minlength=self.n_edges)
            start = np.zeros(self.n_edges + 1, dtype=np.int64)
            np.cumsum(counts, out=start[1:])
            order = np.argsort(self._conn_edges, kind="stable")
            cached = (start.tolist(), self._conn_from[order].tolist())
            self._reverse_cache = cached
        return cached

    # ------------------------------------------------------------------
    # basic accessors
    # ------------------------------------------------------------------
    def node_index(self, node_id: str) -> int:
        return self._index_of_node_id[node_id]

    def out_edges(self, node: int) -> np.ndarray:
        """Indices of the edges leaving ``node``."""
        return self._out[self._out_start[node] : self._out_start[node + 1]]

    def in_edges(self, node: int) -> np.ndarray:
        """Indices of the edges entering ``node``."""
        return self._in[self._in_start[node] : self._in_start[node + 1]]

    def successors(self, node: int) -> np.ndarray:
        """Indices of the nodes reachable from ``node`` in one hop."""
        edges = self.out_edges(node)
        return np.fromiter((self.edges[int(e)].to_node for e in edges), dtype=np.int32)

    def predecessors(self, node: int) -> np.ndarray:
        """Indices of the nodes that reach ``node`` in one hop."""
        edges = self.in_edges(node)
        return np.fromiter((self.edges[int(e)].from_node for e in edges), dtype=np.int32)

    def edge_between(self, tail: int, head: int) -> int | None:
        """Index of the edge ``tail -> head``, or ``None`` when not connected."""
        return self._lookup.get((tail, head))

    def has_edge(self, tail: int, head: int) -> bool:
        return (tail, head) in self._lookup

    def free_flow_time(self) -> np.ndarray:
        """Per-edge free-flow travel time in seconds."""
        return self._free_flow.copy()

    def lengths(self) -> np.ndarray:
        return self._length.copy()

    def speeds(self) -> np.ndarray:
        return self._speed.copy()

    def tails(self) -> np.ndarray:
        """Per-edge index of the junction the edge leaves."""
        return self._tail

    def heads(self) -> np.ndarray:
        """Per-edge index of the junction the edge arrives at."""
        return self._head

    def coordinates(self) -> np.ndarray:
        cached = self._coord_cache
        if cached is None:
            cached = np.array([(n.x, n.y) for n in self.nodes], dtype=np.float64)
            self._coord_cache = cached
        return cached

    def lonlat(self) -> np.ndarray:
        """Per-node ``(lon, lat)``; zeros when the layout has no georeference."""
        return np.array(
            [(n.lon or 0.0, n.lat or 0.0) for n in self.nodes], dtype=np.float64
        )

    def has_georeference(self) -> bool:
        return self.nodes[0].lat is not None if self.nodes else False

    def bounding_box(self) -> tuple[float, float, float, float]:
        coords = self.coordinates()
        return (
            float(coords[:, 0].min()),
            float(coords[:, 1].min()),
            float(coords[:, 0].max()),
            float(coords[:, 1].max()),
        )

    # ------------------------------------------------------------------
    # path evaluation
    # ------------------------------------------------------------------
    def path_edges(self, node_path: Sequence[int]) -> list[int]:
        """Edge indices realising a node sequence; raises when it is not a walk."""
        edges: list[int] = []
        for tail, head in zip(node_path, node_path[1:], strict=False):
            edge = self.edge_between(tail, head)
            if edge is None:
                raise ValueError(f"no edge {tail} -> {head}")
            edges.append(edge)
        return edges

    def path_cost(self, node_path: Sequence[int], cost: np.ndarray) -> float:
        """Total ``cost`` of traversing a node sequence."""
        if len(node_path) < 2:
            return 0.0
        idx = self.path_edges(node_path)
        return float(cost[idx].sum())

    def path_free_flow_time(self, node_path: Sequence[int]) -> float:
        return self.path_cost(node_path, self._free_flow)

    def expand(self, node_path: Sequence[int]) -> list[int]:
        """Turn a node sequence into the concrete edge sequence SUMO needs."""
        return self.path_edges(node_path)

    def nodes_from_edges(self, edge_path: Sequence[int]) -> tuple[int, ...]:
        """Turn an edge sequence back into the node walk it realises.

        The inverse of :meth:`path_edges`; the caller is responsible for having
        produced a contiguous edge sequence.
        """
        if not len(edge_path):
            return ()
        nodes = [self.edges[int(edge_path[0])].from_node]
        nodes.extend(self.edges[int(e)].to_node for e in edge_path)
        return tuple(nodes)

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "conn_start": self._conn_start.tolist(),
            "conn_edges": self._conn_edges.tolist(),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: dict) -> RoadNetwork:
        nodes = [Node(**n) for n in payload["nodes"]]
        edges = [Edge(**e) for e in payload["edges"]]
        if "conn_start" in payload:
            start = payload["conn_start"]
            flat = payload["conn_edges"]
            connections = [flat[start[i] : start[i + 1]] for i in range(len(start) - 1)]
            return cls(nodes, edges, connections)
        return cls(nodes, edges)

    @classmethod
    def load(cls, path: str | Path) -> RoadNetwork:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    # metrics used throughout the reports
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        coords = self.coordinates()
        return {
            "nodes": self.n_nodes,
            "edges": self.n_edges,
            "connections": self.n_connections,
            "turn_restricted": self.has_turn_graph,
            "traffic_lights": sum(1 for n in self.nodes if n.has_traffic_light),
            "total_length_km": round(float(self._length.sum()) / 1000.0, 3),
            "mean_edge_length_m": round(float(self._length.mean()) if self.n_edges else 0.0, 2),
            "mean_speed_kmh": round(
                float(self._speed.mean()) * 3.6 if self.n_edges else 0.0, 2
            ),
            "total_lanes": sum(e.lanes for e in self.edges),
            "bounds": [
                round(v, 1)
                for v in (
                    float(coords[:, 0].min()),
                    float(coords[:, 1].min()),
                    float(coords[:, 0].max()),
                    float(coords[:, 1].max()),
                )
            ],
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RoadNetwork(nodes={self.n_nodes}, edges={self.n_edges})"


def nodes_in_radius(
    network: RoadNetwork, centre: int, radius: float, limit: int | None = None
) -> list[int]:
    """Nodes within ``radius`` metres of ``centre``, nearest first."""
    coords = network.coordinates()
    d = np.hypot(coords[:, 0] - coords[centre, 0], coords[:, 1] - coords[centre, 1])
    order = np.argsort(d)
    picked = [int(i) for i in order if 0 < d[i] <= radius]
    return picked[:limit] if limit else picked


def road_classes(network: RoadNetwork) -> dict[str, int]:
    counts: dict[str, int] = {}
    for edge in network.edges:
        counts[edge.road_class] = counts.get(edge.road_class, 0) + 1
    return counts


def iter_traffic_lights(network: RoadNetwork) -> Iterable[int]:
    for node in network.nodes:
        if node.has_traffic_light:
            yield node.index
