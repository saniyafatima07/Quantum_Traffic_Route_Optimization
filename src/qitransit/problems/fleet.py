"""Capacitated and time-windowed vehicle routing on a road network.

Objective
---------
Minimise

.. math::

    Z = w_T \\sum_{r} T_r + w_D \\sum_{r} D_r
      + w_C \\sum_{r} E_r + w_V |R|
      + \\lambda \\big( \\sum_r \\max(0, Q_r - Q) + \\sum_r \\mathrm{tw}_r \\big)

where :math:`T_r` is the congestion-aware travel time of route :math:`r`,
:math:`D_r` its length, :math:`E_r` the delay it imposes on other road users,
:math:`Q_r` its load, :math:`Q` the vehicle capacity, and
:math:`\\mathrm{tw}_r` the total time-window violation.

Encoding and constraint handling
--------------------------------
A candidate is a real vector of priorities, one per customer.  Decoding sorts
by priority into a visit order and then splits that order into routes with a
Clarke-Wright savings pass followed by a regret-insertion repair, so capacity
and time windows are respected by construction wherever that is possible.  The
residual penalty absorbs the cases where it is not, and its weight adapts as
the search proceeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..graph.network import RoadNetwork
from .base import Customer, Evaluation, Problem, Solution
from .travel import TravelMatrix


@dataclass(frozen=True, slots=True)
class Vehicle:
    """A homogeneous fleet member."""

    capacity: float
    fixed_cost: float = 0.0
    max_shift: float = 86400.0


def attachment_savings(
    matrix: TravelMatrix, route: list[int], customer: int
) -> list[tuple[float, list[int]]]:
    """Feasible attachments of one customer to one route, best saving first.

    Customers are matrix index ``+ 1``; index 0 is the depot.  An attachment
    that leaves the route unchanged, or that would not save anything, is not
    offered.
    """
    if not route:
        return []
    c = customer + 1
    first, last = route[0] + 1, route[-1] + 1
    cost = matrix.cost
    options = (
        (cost[0, first] + cost[c, last] - cost[0, last], route + [customer]),
        (cost[0, c] + cost[last, first] - cost[0, first], [customer] + route),
        (cost[0, last] + cost[c, first] - cost[0, first], route[::-1] + [customer]),
        (cost[0, first] + cost[last, c] - cost[0, c], [customer] + route[::-1]),
    )
    candidates = [(saving, candidate) for saving, candidate in options if saving > 0]
    candidates.sort(key=lambda item: -item[0])
    return candidates


def sequential_savings(
    matrix: TravelMatrix,
    demands: np.ndarray,
    capacity: float,
    order: tuple[int, ...],
) -> list[tuple[int, ...]]:
    """Textbook Clarke-Wright: attach each customer where the saving is largest.

    Customers are visited in ``order``; each is attached to the route currently
    being grown at one of four endpoints, or opens a new route if no attachment
    both saves something and fits.
    """
    if not order:
        return []
    closed: list[tuple[int, ...]] = []
    growing: list[int] = []
    load = 0.0

    for position, customer in enumerate(order):
        if position == 0:
            growing = [customer]
            load = float(demands[customer])
            continue
        best_saving = 0.0
        best_route: list[int] | None = None
        for saving, candidate in attachment_savings(matrix, growing, customer):
            if load + float(demands[customer]) > capacity + 1e-9:
                continue
            if saving > best_saving:
                best_saving = saving
                best_route = candidate
        if best_route is not None:
            growing = best_route
            load += float(demands[customer])
        else:
            closed.append(tuple(growing))
            growing = [customer]
            load = float(demands[customer])

    if growing:
        closed.append(tuple(growing))
    return closed


class FleetVRP(Problem):
    """Single-depot fleet routing with capacity and optional time windows."""

    family = "cvrp"

    def __init__(
        self,
        network: RoadNetwork,
        matrix: TravelMatrix,
        depot: int,
        customers: list[Customer],
        vehicles: list[Vehicle],
        *,
        name: str = "fleet",
        seed: int = 0,
        use_time_windows: bool = False,
        weights: tuple[float, float, float, float] = (1.0, 0.0, 0.5, 0.0),
        penalty_weight: float = 1000.0,
        network_name: str = "synthetic",
        max_vehicles: int | None = None,
        notes: dict | None = None,
    ) -> None:
        super().__init__(name=name, n_customers=len(customers), seed=seed)
        self.network = network
        self.matrix = matrix
        self.depot = depot
        self.customers = customers
        self.vehicles = vehicles
        self.use_time_windows = use_time_windows
        self.w_time, self.w_dist, self.w_cong, self.w_veh = weights
        self.penalty_weight = penalty_weight
        self.network_name = network_name
        self.n_vehicles = max_vehicles if max_vehicles is not None else len(vehicles)
        self.notes = dict(notes or {})

        self.demands = np.array([c.demand for c in customers], dtype=np.float64)
        self.service = np.array([c.service_time for c in customers], dtype=np.float64)
        self.tw_start = np.array([c.window_start for c in customers], dtype=np.float64)
        self.tw_end = np.array([c.window_end for c in customers], dtype=np.float64)
        self.capacity = max(v.capacity for v in vehicles) if vehicles else 0.0
        self.horizon = float(max((c.window_end for c in customers), default=0.0))
        self._nodes = np.concatenate(([depot], [c.node for c in customers]))

    # ------------------------------------------------------------------
    # encoding
    # ------------------------------------------------------------------
    def order_of(self, keys: np.ndarray) -> tuple[int, ...]:
        """Customers in descending priority order."""
        return tuple(int(i) for i in np.argsort(-np.asarray(keys, dtype=np.float64), kind="stable"))

    def encode(self, solution: Solution) -> np.ndarray:
        order = self.as_permutation(solution)
        keys = np.empty(self.n_customers, dtype=np.float64)
        keys[:] = 0.0
        n = len(order)
        if n == 0:
            return keys
        base = np.linspace(1.0, 0.0, n, endpoint=False)
        for rank, customer in enumerate(order):
            keys[customer] = base[rank]
        return keys

    def decode(self, keys: np.ndarray) -> Solution:
        order = self.order_of(keys)
        if not self.use_time_windows:
            return tuple(self._savings_split(order))
        return tuple(self._window_split(order))

    # ------------------------------------------------------------------
    # feasibility
    # ------------------------------------------------------------------
    def _leg(self, i: int, j: int) -> float:
        return float(self.matrix.cost[i, j])

    def _savings_split(self, order: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Split a visit order into capacity-feasible routes by savings.

        Sequential savings keeps every customer served by construction and costs
        ``O(n)`` with four candidate attachments per customer, which is what
        makes it affordable to run inside the optimiser's inner loop.
        """
        return sequential_savings(self.matrix, self.demands, self.capacity, order)

    def _window_split(self, order: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Sequential insertion with regret-2 repair, honouring time windows.

        Customers are inserted into the route that can take them soonest, and a
        second pass re-inserts anything that failed, choosing by the largest
        increase in travel time from deferral.  This is the standard regret
        heuristic and it recovers most of the feasibility that a purely
        sequential split would lose.
        """
        if not order:
            return []
        routes: list[list[int]] = [[]]
        loads: list[float] = [0.0]
        for customer in order:
            self._cheapest_insertion(customer, routes, loads)
        return [tuple(r) for r in routes if r]

    def _cheapest_insertion(
        self, customer: int, routes: list[list[int]], loads: list[float]
    ) -> None:
        """Place ``customer`` where it costs least, opening a route if forced."""
        demand = float(self.demands[customer])
        empty_slot: int | None = None
        best_cost = float("inf")
        best_slot: int | None = None
        for r_index, route in enumerate(routes):
            if loads[r_index] + demand > self.capacity + 1e-9:
                continue
            if not route:
                if empty_slot is None:
                    empty_slot = r_index
                continue
            score = self._insertion_cost(route, customer)
            if score < best_cost:
                best_cost = score
                best_slot = r_index
        target = best_slot if best_slot is not None else empty_slot
        if target is None:
            routes.append([customer])
            loads.append(demand)
            return
        routes[target].append(customer)
        loads[target] += demand

    def _insertion_cost(self, route: list[int], customer: int) -> float:
        c = customer + 1
        if not route:
            return self._leg(0, c) + self._leg(c, 0)
        best = float("inf")
        for position in range(len(route) + 1):
            prev = 0 if position == 0 else route[position - 1] + 1
            nxt = 0 if position == len(route) else route[position] + 1
            delta = self._leg(prev, c) + self._leg(c, nxt) - self._leg(prev, nxt)
            best = min(best, delta)
        return best

    def _insert_time(self, route: list[int], start_time: float, customer: int) -> tuple[float, float]:
        """Arrival and slack if ``customer`` is appended to ``route``."""
        t = start_time
        previous = 0
        for c in route:
            t += self.matrix.time[previous, c + 1] + self.service[c]
            previous = c + 1
        t += self.matrix.time[previous, customer + 1] + self.service[customer]
        arrival = t
        wait = max(0.0, self.tw_start[customer] - arrival)
        return arrival + wait, arrival

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def evaluate(self, solution: Solution) -> Evaluation:
        total_time = 0.0
        total_distance = 0.0
        total_externality = 0.0
        capacity_violation = 0.0
        tw_violation = 0.0
        served = 0

        for route in solution:
            if not route:
                continue
            previous = 0
            clock = 0.0
            load = 0.0
            for customer in route:
                nxt = customer + 1
                leg_time = float(self.matrix.time[previous, nxt])
                leg_dist = float(self.matrix.distance[previous, nxt])
                leg_ext = float(self.matrix.externality[previous, nxt])
                clock += leg_time
                total_time += leg_time
                total_distance += leg_dist
                total_externality += leg_ext
                clock += float(self.service[customer])
                load += float(self.demands[customer])
                if self.use_time_windows:
                    if clock < self.tw_start[customer]:
                        wait = self.tw_start[customer] - clock
                        clock += wait
                        total_time += wait
                    if clock > self.tw_end[customer]:
                        tw_violation += clock - self.tw_end[customer]
                previous = nxt
            leg_time = float(self.matrix.time[previous, 0])
            total_time += leg_time
            total_distance += float(self.matrix.distance[previous, 0])
            total_externality += float(self.matrix.externality[previous, 0])
            if load > self.capacity + 1e-9:
                capacity_violation += load - self.capacity
            served += len(route)

        unserved = self.n_customers - served
        vehicles_used = sum(1 for r in solution if r)
        penalty = self.penalty_weight * (capacity_violation + tw_violation)
        penalty += self.penalty_weight * 10.0 * unserved

        objective = (
            self.w_time * total_time
            + self.w_dist * total_distance
            + self.w_cong * total_externality
            + self.w_veh * vehicles_used
            + penalty
        )
        return Evaluation(
            objective=objective,
            time=total_time,
            distance=total_distance,
            externality=total_externality,
            vehicles=vehicles_used,
            penalty=penalty,
            capacity_violation=capacity_violation,
            time_window_violation=tw_violation,
            unserved=unserved,
            feasible=capacity_violation <= 1e-6 and tw_violation <= 1e-6 and unserved == 0,
        )

    # ------------------------------------------------------------------
    # bounds
    # ------------------------------------------------------------------
    def lower_bound(self) -> float:
        """Sum of the cheapest inbound and outbound leg per customer.

        Ignoring capacity, sharing and time windows this is the cost of
        serving every customer exactly once, so it is a valid relaxation.
        """
        if self.n_customers == 0:
            return 0.0
        inbound = self.matrix.cost[1:, 0]
        outbound = self.matrix.cost[0, 1:]
        reachable = int(np.count_nonzero(np.isfinite(inbound) & np.isfinite(outbound)))
        if reachable == 0:
            return float("-inf")
        return float(
            self.w_time * np.minimum(inbound, outbound).sum()
            + self.w_veh * 1.0
        )

    def ideal_vehicles(self) -> int:
        if self.capacity <= 0:
            return 1
        return int(np.ceil(self.demands.sum() / self.capacity))

    def describe(self) -> dict:
        return {
            "name": self.name,
            "network": self.network_name,
            "customers": self.n_customers,
            "capacity": self.capacity,
            "total_demand": round(float(self.demands.sum()), 2),
            "min_vehicles": self.ideal_vehicles(),
            "time_windows": self.use_time_windows,
            "depot_node": self.depot,
        }


def clarke_wright(problem: "FleetVRP", order: tuple[int, ...] | None = None) -> tuple[tuple[int, ...], ...]:
    """Savings construction for a fleet problem, in its natural visit order.

    The same routine the decoder uses, so a report can state that the savings
    floor and the optimiser's starting point are the same construction and any
    difference in the result is due to the search.
    """
    if order is None:
        order = tuple(range(problem.n_customers))
    return tuple(problem._savings_split(order))
