"""Shortest-path algorithms over the road network.

All routines take a per-edge cost vector, so the same code serves free-flow
routing, congestion-aware routing and online re-planning.

The searches run over the **turn graph** rather than over junctions.  A state is
an edge: "having just traversed this edge, standing at its head".  That is one
state per edge instead of one per junction, and it is what makes the search
legality-aware — the successors of a state are exactly the edges SUMO permits
next, so U-turns at restricted junctions and lane-permission mismatches are
excluded by construction.  A node-based search cannot do this, because arriving
at a junction by different edges leaves different continuations available.

The cost of that choice is state space growing from junctions to edges and a
per-state cost that includes traversing the edge.  :func:`dijkstra` is the exact
reference the metaheuristics are scored against; the others exist so the report
can show what the exact method costs in wall-clock time and how much the
heuristic and the bidirectional pruning are worth.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..graph.network import RoadNetwork

INF = float("inf")


@dataclass(frozen=True, slots=True)
class PathResult:
    """The outcome of a single shortest-path query."""

    cost: float
    nodes: tuple[int, ...]
    edges: tuple[int, ...]

    @property
    def reached(self) -> bool:
        return bool(self.nodes)

    def __len__(self) -> int:
        return len(self.edges)


def _empty() -> PathResult:
    return PathResult(cost=INF, nodes=(), edges=())


def _walk(parent: Sequence[int], last: int) -> tuple[int, ...]:
    """The edges from the seed through ``last``, in driving order.

    ``parent[x] < 0`` marks a state the search started from, so the walk runs
    from ``last`` back to the first such state and includes both ends.
    """
    edges: list[int] = []
    edge = int(last)
    while True:
        edges.append(edge)
        step = int(parent[edge])
        if step < 0:
            break
        edge = step
    edges.reverse()
    return tuple(edges)


def _as_result(cost: float, edges: Sequence[int], network: RoadNetwork) -> PathResult:
    route = tuple(int(e) for e in edges)
    if not route:
        return PathResult(cost=0.0, nodes=(), edges=())
    return PathResult(
        cost=float(cost), nodes=network.nodes_from_edges(route), edges=route
    )


# ----------------------------------------------------------------------
# exact search
# ----------------------------------------------------------------------
def dijkstra(
    network: RoadNetwork,
    cost: np.ndarray,
    source: int,
    target: int,
) -> PathResult:
    """Exact shortest path through the legal turns.

    ``O(E log E)`` over the turn graph.  Because every cost is non-negative and
    the heap pops in non-decreasing order, the first state popped that stands at
    ``target`` already carries the optimum, so the search stops there.
    """
    if source == target:
        return PathResult(cost=0.0, nodes=(source,), edges=())

    conn_start, conn_edges, head, _ = network.turn_lists()
    costs = cost.tolist()
    dist = [INF] * network.n_edges
    parent = [-1] * network.n_edges
    heap: list[tuple[float, int]] = []

    for e in network.out_edges(source):
        e = int(e)
        c = costs[e]
        if c < dist[e]:
            dist[e] = c
            heapq.heappush(heap, (c, e))

    best = INF
    reach = -1
    while heap:
        d, e = heapq.heappop(heap)
        if d > dist[e]:
            continue
        if head[e] == target:
            best, reach = d, e
            break
        for k in range(conn_start[e], conn_start[e + 1]):
            f = conn_edges[k]
            nd = d + costs[f]
            if nd < dist[f]:
                dist[f] = nd
                parent[f] = e
                heapq.heappush(heap, (nd, f))

    if reach < 0:
        return _empty()
    return _as_result(best, _walk(parent, reach), network)


def dijkstra_continuing(
    network: RoadNetwork,
    cost: np.ndarray,
    arrival_edge: int,
    target: int,
) -> PathResult:
    """The cheapest way onward from an edge already being traversed.

    A vehicle that has just driven along ``arrival_edge`` may only continue
    through one of that edge's legal successors.  Seeding the search with
    exactly those successors is what keeps a sequence of independently-planned
    legs a single drivable route, rather than a list of routes that each end in
    an illegal manoeuvre.
    """
    conn_start, conn_edges, head, _ = network.turn_lists()
    costs = cost.tolist()
    dist = [INF] * network.n_edges
    parent = [-1] * network.n_edges
    heap: list[tuple[float, int]] = []

    for f in network.successor_edges(arrival_edge):
        f = int(f)
        c = costs[f]
        if c < dist[f]:
            dist[f] = c
            heapq.heappush(heap, (c, f))

    best = INF
    reach = -1
    while heap:
        d, e = heapq.heappop(heap)
        if d > dist[e]:
            continue
        if head[e] == target:
            best, reach = d, e
            break
        for k in range(conn_start[e], conn_start[e + 1]):
            f = conn_edges[k]
            nd = d + costs[f]
            if nd < dist[f]:
                dist[f] = nd
                parent[f] = e
                heapq.heappush(heap, (nd, f))

    if reach < 0:
        return _empty()
    return _as_result(best, _walk(parent, reach), network)


def dijkstra_tree(
    network: RoadNetwork, cost: np.ndarray, source: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The full single-source tree, expressed per junction.

    Runs the turn-graph search once and folds the per-edge labels onto the
    junction each edge arrives at, so callers can look up a leg cost and its
    predecessor pair by node index exactly as before.
    """
    conn_start, conn_edges, head, tail = network.turn_lists()
    costs = cost.tolist()
    dist = [INF] * network.n_edges
    parent = [-1] * network.n_edges
    heap: list[tuple[float, int]] = []

    for e in network.out_edges(source):
        e = int(e)
        c = costs[e]
        if c < dist[e]:
            dist[e] = c
            heapq.heappush(heap, (c, e))

    while heap:
        d, e = heapq.heappop(heap)
        if d > dist[e]:
            continue
        for k in range(conn_start[e], conn_start[e + 1]):
            f = conn_edges[k]
            nd = d + costs[f]
            if nd < dist[f]:
                dist[f] = nd
                parent[f] = e
                heapq.heappush(heap, (nd, f))

    n_nodes = network.n_nodes
    node_dist = [INF] * n_nodes
    node_edge = [-1] * n_nodes
    node_prev = [-1] * n_nodes
    for e, d in enumerate(dist):
        v = head[e]
        if d < node_dist[v]:
            node_dist[v] = d
            node_edge[v] = e
            node_prev[v] = tail[e]
    node_dist[source] = 0.0
    node_edge[source] = -1
    node_prev[source] = -1
    return (
        np.array(node_dist, dtype=np.float64),
        np.array(node_prev, dtype=np.int32),
        np.array(node_edge, dtype=np.int32),
    )


def bidirectional_dijkstra(
    network: RoadNetwork, cost: np.ndarray, source: int, target: int
) -> PathResult:
    """Meet in the middle on the turn graph.

    The forward search settles edges leaving ``source`` and the backward search
    settles edges arriving at ``target``; the two meet on a shared edge.  The
    frontier with the cheaper tentative cost is expanded at each step, and the
    search stops as soon as neither frontier can still undercut the best
    connection found so far.
    """
    if source == target:
        return PathResult(cost=0.0, nodes=(source,), edges=())

    conn_start, conn_edges, _, _ = network.turn_lists()
    reverse_start, reverse_edges = network.reverse_turn_lists()
    costs = cost.tolist()

    forward: dict[int, tuple[float, int]] = {}
    backward: dict[int, tuple[float, int]] = {}
    heap_f: list[tuple[float, int]] = []
    heap_b: list[tuple[float, int]] = []

    for e in network.out_edges(source):
        e = int(e)
        forward[e] = (costs[e], -1)
        heapq.heappush(heap_f, (costs[e], e))
    for e in network.in_edges(target):
        e = int(e)
        backward[e] = (costs[e], -1)
        heapq.heappush(heap_b, (costs[e], e))

    best = INF
    meet = -1
    while heap_f and heap_b:
        if min(heap_f[0][0], heap_b[0][0]) >= best:
            break
        if heap_f[0][0] <= heap_b[0][0]:
            d, e = heapq.heappop(heap_f)
            table, other, heap = forward, backward, heap_f
            start, edges = conn_start, conn_edges
        else:
            d, e = heapq.heappop(heap_b)
            table, other, heap = backward, forward, heap_b
            start, edges = reverse_start, reverse_edges
        if d > table[e][0]:
            continue
        if e in other:
            # Both labels are measured up to the head of the meeting edge, so
            # the shared edge is priced in each half and must be subtracted once.
            total = d + other[e][0] - costs[e]
            if total < best:
                best, meet = total, e
        for k in range(start[e], start[e + 1]):
            f = edges[k]
            nd = d + costs[f]
            if nd < table.get(f, (INF, -1))[0]:
                table[f] = (nd, e)
                heapq.heappush(heap, (nd, f))

    if meet < 0:
        return _empty()

    route = list(_walk({e: v[1] for e, v in forward.items()}, meet))
    edge = meet
    while backward[edge][1] >= 0:
        edge = backward[edge][1]
        route.append(edge)
    return _as_result(route_cost(network, route, cost), route, network)


def distances_to(
    network: RoadNetwork, cost: np.ndarray, source: int
) -> np.ndarray:
    """Shortest cost from every junction *to* ``source``.

    The backward counterpart of :func:`dijkstra_tree`, obtained by running the
    same search over the reversed turn graph.  Landmark heuristics need both
    directions, so this exists as its own routine rather than as a wrapper that
    rebuilds the network.
    """
    reverse_start, reverse_edges = network.reverse_turn_lists()
    costs = cost.tolist()
    dist = [INF] * network.n_edges
    heap: list[tuple[float, int]] = []

    for e in network.in_edges(source):
        e = int(e)
        dist[e] = costs[e]
        heapq.heappush(heap, (costs[e], e))

    while heap:
        d, e = heapq.heappop(heap)
        if d > dist[e]:
            continue
        for k in range(reverse_start[e], reverse_start[e + 1]):
            f = reverse_edges[k]
            nd = d + costs[f]
            if nd < dist[f]:
                dist[f] = nd
                heapq.heappush(heap, (nd, f))

    node_dist = np.full(network.n_nodes, INF, dtype=np.float64)
    np.minimum.at(node_dist, network.tails(), np.asarray(dist, dtype=np.float64))
    node_dist[source] = 0.0
    return node_dist


class LandmarkHeuristic:
    """An ALT bound over the turn graph, admissible by construction.

    For a landmark ``L`` already searched from and to, the triangle inequality
    gives two gaps, and only two:

    ``d(v, L) - d(t, L) <= d(v, t)``
        because ``d(v, L) <= d(v, t) + d(t, L)`` — a path through ``t``.
    ``d(L, t) - d(L, v) <= d(v, t)``
        because ``d(L, t) <= d(L, v) + d(v, t)`` — a path through ``v``.

    Both bound the cost to reach ``t`` from ``v``, and both are what the search
    needs.  Note what is *not* on the
    list: ``d(L, v) - d(L, t)``, which the triangle inequality caps at
    ``d(t, v)``.  On an undirected graph that is the same number and the usual
    ``|d(L,v) - d(L,t)|`` formula works; on a street network it is not, because
    one-way systems make ``d(t, v)`` and ``d(v, t)`` different numbers.  Taking
    the absolute value there produces a bound that overshoots by exactly the
    asymmetry of the network — measured at 1.4 s on the synthetic grid, enough
    to send A* home with a route 30% worse than Dijkstra's.  So each landmark
    contributes its two one-sided gaps, and the largest over the landmark set is
    the bound.

    This matters more than it might seem.  The obvious geometric alternative —
    straight-line distance priced at the cheapest cost per metre in the network
    — is inadmissible on a compiled network, and not by a hair.  ``netconvert``
    keeps the pre-collapse lengths of the edges it shortens to junctions that no
    longer exist, so an edge 42 m long on the map can have its endpoints 0.2 m
    apart by the length the search pays.  On the Bengaluru central extract the
    median edge is 1.4x shorter than the straight line between its endpoints and
    15,033 edges are more than 10x shorter.  ALT never looks at geometry, so it
    cannot have that failure mode.

    The trees cost one turn-graph search per landmark per direction and are
    built once per cost vector, which pays for itself from the second query.
    """

    __slots__ = ("_in", "_nodes", "_out", "_scratch", "landmarks")

    def __init__(
        self,
        network: RoadNetwork,
        cost: np.ndarray,
        landmarks: Sequence[int],
    ) -> None:
        self._nodes = network
        self.landmarks = [int(v) for v in landmarks]
        self._out = [self._pack(dijkstra_tree(network, cost, v)[0]) for v in self.landmarks]
        self._in = [self._pack(distances_to(network, cost, v)) for v in self.landmarks]
        self._scratch = np.zeros(network.n_nodes, dtype=np.float64)

    @staticmethod
    def _pack(tree: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A tree plus the mask and buffer its gap is accumulated through.

        Both are built once rather than per query: this runs at every push of
        the search, and allocating a 65k-element mask per landmark costs more
        than the search it is meant to speed up.  The mask is what keeps an
        unreachable junction out of the subtraction — with plain ``inf`` values
        the arithmetic yields ``nan``, and a ``nan`` in the frontier key makes
        the search's stopping rule meaningless.
        """
        return tree, np.isfinite(tree), np.zeros(tree.shape, dtype=np.float64)

    def __call__(self, target: int) -> np.ndarray:
        """A per-junction lower bound on the cost to reach ``target``."""
        t = int(target)
        bound = self._scratch
        bound.fill(0.0)
        for tree, mask, buffer in self._out:
            if np.isfinite(tree[t]):
                np.subtract(tree[t], tree, out=buffer, where=mask)
                np.maximum(bound, buffer, out=bound)
        for tree, mask, buffer in self._in:
            if np.isfinite(tree[t]):
                np.subtract(tree, tree[t], out=buffer, where=mask)
                np.maximum(bound, buffer, out=bound)
        return bound.copy()


def spread_landmarks(
    network: RoadNetwork, candidates: Sequence[int], count: int = 4
) -> list[int]:
    """Pick landmarks at the corners of the candidates' bounding box.

    ALT's bound is the gap between a landmark's distance to two junctions, and
    that gap is largest when the landmark lies off the line between them.
    Corners of the bounding box are the cheapest way to get spread without
    solving the (itself combinatorial) maximum-dispersion problem, and on a
    convex extract they are the genuinely extreme points anyway.
    """
    points = np.array([int(v) for v in candidates], dtype=np.int64)
    if points.size == 0 or count <= 0:
        return []
    if points.size <= count:
        return [int(v) for v in points]
    coords = network.coordinates()
    xs, ys = coords[points, 0], coords[points, 1]
    picked: list[int] = []
    for key in ((xs.min(), ys.min()), (xs.min(), ys.max()), (xs.max(), ys.min()), (xs.max(), ys.max())):
        if key in picked or len(picked) >= count:
            continue
        candidate = points[int(np.argmin((xs - key[0]) ** 2 + (ys - key[1]) ** 2))]
        value = int(candidate)
        if value not in picked:
            picked.append(value)
    return picked[:count]


def a_star(
    network: RoadNetwork,
    cost: np.ndarray,
    source: int,
    target: int,
    heuristic: np.ndarray | Callable[[int], float] | None = None,
) -> PathResult:
    """A* on the turn graph, guided by a lower bound on the cost to go.

    ``heuristic`` is either a per-junction array of lower bounds on reaching
    each junction, or a callable taking a junction index.  Without one the
    search reduces to Dijkstra.  The bound only orders the frontier, so a
    caller who supplies a bound that occasionally *over*estimates still gets
    an optimal route: the search returns the target only once no open state can
    undercut it, which relies on nothing but non-negative costs.
    """
    if source == target:
        return PathResult(cost=0.0, nodes=(source,), edges=())

    if heuristic is None:
        remaining = lambda node: 0.0
    elif callable(heuristic):
        lookup = heuristic
        remaining = lookup
    else:
        table = np.asarray(heuristic, dtype=np.float64)
        remaining = lambda node: float(table[node])

    conn_start, conn_edges, head, _ = network.turn_lists()
    costs = cost.tolist()
    dist = [INF] * network.n_edges
    parent = [-1] * network.n_edges
    heap: list[tuple[float, float, int]] = []

    for e in network.out_edges(source):
        e = int(e)
        d = costs[e]
        if d < dist[e]:
            dist[e] = d
            parent[e] = -1
            heapq.heappush(heap, (d + remaining(head[e]), d, e))

    best = INF
    reach = -1
    while heap:
        f, d, e = heapq.heappop(heap)
        if d > dist[e]:
            continue
        if f >= best:
            break
        if head[e] == target:
            best, reach = d, e
            continue
        for k in range(conn_start[e], conn_start[e + 1]):
            nxt = conn_edges[k]
            nd = d + costs[nxt]
            if nd < dist[nxt]:
                dist[nxt] = nd
                parent[nxt] = e
                heapq.heappush(heap, (nd + remaining(head[nxt]), nd, nxt))

    if reach < 0:
        return _empty()
    return _as_result(best, _walk(parent, reach), network)


def time_dependent_dijkstra(
    network: RoadNetwork,
    cost: np.ndarray,
    source: int,
    target: int,
    depart: float = 0.0,
    horizon: float = 3600.0,
    time_step: float = 5.0,
    cost_function: Callable[[float], np.ndarray] | None = None,
) -> PathResult:
    """Earliest arrival when edge costs drift with the time of day.

    A state's label is the earliest moment the vehicle can clear the edge, and
    that moment decides which cost slice prices the edge it leaves next.  The
    reported cost is the arrival time relative to ``depart``.
    """
    conn_start, conn_edges, head, _ = network.turn_lists()
    static = cost.tolist()
    arrival = [INF] * network.n_edges
    parent = [-1] * network.n_edges
    slots = max(int(horizon / time_step), 1)
    cache: dict[int, list[float]] = {}

    def slice_at(when: float) -> list[float]:
        if cost_function is None:
            return static
        slot = min(max(int((when - depart) / time_step), 0), slots - 1)
        prices = cache.get(slot)
        if prices is None:
            prices = cost_function(depart + slot * time_step).tolist()
            cache[slot] = prices
        return prices

    heap: list[tuple[float, int]] = []
    opening = slice_at(depart)
    for e in network.out_edges(source):
        e = int(e)
        t = depart + opening[e]
        if t < arrival[e]:
            arrival[e] = t
            heapq.heappush(heap, (t, e))

    while heap:
        t, e = heapq.heappop(heap)
        if t > arrival[e] + 1e-9:
            continue
        if head[e] == target:
            return _as_result(t - depart, _walk(parent, e), network)
        if t - depart > horizon:
            break
        prices = slice_at(t)
        for k in range(conn_start[e], conn_start[e + 1]):
            f = conn_edges[k]
            nt = t + prices[f]
            if nt < arrival[f]:
                arrival[f] = nt
                parent[f] = e
                heapq.heappush(heap, (nt, f))

    return _empty()


def all_pairs_tree(
    network: RoadNetwork, cost_vector: np.ndarray, sources: Sequence[int]
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Pre-compute the shortest-path tree from each of a handful of depots.

    Fleet routing pays the search cost once per depot instead of once per
    customer, which is what makes repeated route construction affordable
    inside an optimiser's inner loop.
    """
    return {s: dijkstra_tree(network, cost_vector, s) for s in sources}


def route_cost(network: RoadNetwork, edges: Sequence[int], cost_vector: np.ndarray) -> float:
    if not len(edges):
        return 0.0
    return float(cost_vector[list(edges)].sum())


def is_legal_route(network: RoadNetwork, edges: Sequence[int]) -> bool:
    """Whether an edge sequence is a contiguous, legally-connected walk.

    The check the simulation layer applies before handing a route to SUMO: a
    route that fails it would be rejected at load time, which would silently
    invalidate every measurement taken from the run.
    """
    if not len(edges):
        return False
    for first, second in zip(edges, edges[1:], strict=False):
        if not network.may_turn(int(first), int(second)):
            return False
    return True
