"""SUMO demand generation and simulation driving.

This is the layer that answers the question the optimisers cannot answer on their
own: *does the route the optimiser produced actually work on the real network?*
A solution is converted to vehicle routes, run through SUMO against the compiled
Bengaluru network, and measured in the terms a fleet manager cares about —
realised travel time, distance, time lost to congestion, and emissions.

Simulation is deliberately kept out of the optimiser's inner loop: a TraCI step
costs milliseconds, so evaluating thousands of candidates inside it would
dominate the run.  It is the *validation* stage, applied to the handful of
solutions the benchmark actually produces.

No ``.sumocfg`` is used: SUMO 1.26 rejects them in this configuration, and the
command line is what the platform needs to control anyway.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import quoteattr

import numpy as np

from ..graph.network import RoadNetwork
from ..routing.shortest_path import dijkstra, dijkstra_continuing
from ..sumo_env import import_traci, sumo_binary, sumo_version


@dataclass(frozen=True, slots=True)
class VehicleProfile:
    """A vehicle type: the physical parameters that set its cost and emissions.

    All three profiles share the ``delivery`` vehicle class.  That is the honest
    classification for the problem, and it is also the only class every lane of
    the OSM import permits: several Bengaluru service lanes are restricted to
    ``delivery``, so a mixed fleet refuses to stop on them while the routing
    model — which treats every edge as usable — happily routes over them.  The
    profiles still differ in the dynamics that actually generate congestion:
    acceleration, deceleration, length, and the gap that follows.

    ``emission_class`` is left unset: SUMO's ``vClass``-derived emission model is
    both valid and the one the congestion literature reports against, while
    naming an emission class explicitly makes the routes file fail on stock
    installs.
    """

    name: str
    accel: float = 2.6
    decel: float = 4.5
    sigma: float = 0.5
    length: float = 5.0
    min_gap: float = 2.5
    max_speed: float = 55.5
    v_class: str = "delivery"

    def as_xml(self) -> str:
        fields = {
            "id": self.name,
            "accel": f"{self.accel}",
            "decel": f"{self.decel}",
            "sigma": f"{self.sigma}",
            "length": f"{self.length}",
            "minGap": f"{self.min_gap}",
            "maxSpeed": f"{self.max_speed}",
            "vClass": self.v_class,
        }
        rendered = " ".join(f"{key}={quoteattr(value)}" for key, value in fields.items())
        return f"    <vType {rendered}/>"


DELIVERY_VAN = VehicleProfile("deliveryVan", length=7.0)
CARGO_TRUCK = VehicleProfile("cargoTruck", length=9.0, accel=2.0)
AUTO_RICKSHAW = VehicleProfile("autoRickshaw", length=3.5, accel=3.5)

BENGALURU_FLEET = (DELIVERY_VAN, CARGO_TRUCK, AUTO_RICKSHAW)


@dataclass(frozen=True, slots=True)
class RouteAssignment:
    """One vehicle's itinerary, in the form SUMO will simulate."""

    vehicle_id: str
    vehicle_type: str
    edge_ids: tuple[str, ...]
    depart: float
    stop_lanes: tuple[str, ...] = ()
    service_time: float = 240.0

    def as_xml(self) -> str:
        """A ``<vehicle>`` element with an explicit edge route and its stops.

        Stops are emitted with no ``endPos``: SUMO then halts the vehicle at the
        end of the named lane, which is where a delivery happens on an edge-based
        route, and omitting it avoids depending on which position specifiers a
        given SUMO build accepts.
        """
        parts = [
            f"    <vehicle id={quoteattr(self.vehicle_id)} "
            f"type={quoteattr(self.vehicle_type)} depart={quoteattr(f'{self.depart:.2f}')} "
            'departLane="free" departSpeed="max">',
            f"        <route edges={quoteattr(' '.join(self.edge_ids))}/>",
        ]
        for lane in self.stop_lanes:
            parts.append(
                f"        <stop lane={quoteattr(lane)} "
                f"duration={quoteattr(f'{self.service_time:.1f}')}/>"
            )
        parts.append("    </vehicle>")
        return "\n".join(parts)


def write_routes(
    path: str | Path,
    assignments: list[RouteAssignment],
    profiles: tuple[VehicleProfile, ...] = BENGALURU_FLEET,
) -> Path:
    """Write a ``.rou.xml`` with explicit vehicle types and edge routes.

    SUMO needs the ``<vType>`` declared in the routes file for TraCI to have
    anything to attach a vehicle to, and edge ids rather than node ids so the
    vehicle follows exactly the edges the optimiser priced.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
    ]
    lines.extend(profile.as_xml() for profile in profiles)
    lines.extend(a.as_xml() for a in assignments if a.edge_ids)
    lines.append("</routes>")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def build_assignments(
    network: RoadNetwork,
    legs: list[tuple[int, ...]],
    service_positions: dict[int, tuple[int, ...]] | None = None,
    profiles: tuple[VehicleProfile, ...] = BENGALURU_FLEET,
    departs: np.ndarray | None = None,
    service_time: float = 240.0,
) -> list[RouteAssignment]:
    """Turn per-vehicle edge legs into route assignments.

    ``service_positions`` maps a vehicle's position in ``legs`` to positions
    within its own edge list where a delivery happens; each becomes a stop on
    the first lane of that edge.  The lane index is read from the edge rather
    than assumed, because a stop naming a lane the edge does not have makes SUMO
    reject the whole routes file.
    """
    if departs is None:
        departs = np.zeros(len(legs), dtype=np.float64)
    assignments: list[RouteAssignment] = []
    for index, edges in enumerate(legs):
        if not edges:
            continue
        profile = profiles[index % len(profiles)]
        stops = (service_positions or {}).get(index, ())
        lanes = tuple(
            f"{network.edges[edges[p]].id}_0"
            for p in stops
            if 0 <= p < len(edges) and network.edges[edges[p]].lanes > 0
        )
        assignments.append(
            RouteAssignment(
                vehicle_id=f"{profile.name}_{index}",
                vehicle_type=profile.name,
                edge_ids=tuple(network.edges[e].id for e in edges),
                depart=float(departs[index % departs.size]),
                stop_lanes=lanes,
                service_time=service_time,
            )
        )
    return assignments


def departure_schedule(n_vehicles: int, window: float = 1800.0, seed: int = 0) -> np.ndarray:
    """Departure times spread over a peak window with a reproducible jitter."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0.0, window, max(n_vehicles, 1), endpoint=False)
    return base + rng.uniform(0.0, window / max(n_vehicles, 1), size=base.size)


@dataclass
class SimulationReport:
    """What a SUMO run actually measured."""

    network: str
    vehicles: int
    departures: int
    arrivals: int
    finished_vehicles: int
    mean_travel_time: float
    mean_waiting_time: float
    mean_distance: float
    total_time_loss: float
    total_co2_g: float
    mean_network_speed: float
    completion_rate: float
    wall_time: float
    steps: int
    seed: int
    per_vehicle: dict[str, dict] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "network": self.network,
            "vehicles": self.vehicles,
            "departures": self.departures,
            "arrivals": self.arrivals,
            "finished_vehicles": self.finished_vehicles,
            "completion_rate": round(self.completion_rate, 4),
            "mean_travel_time_s": round(self.mean_travel_time, 2),
            "mean_waiting_time_s": round(self.mean_waiting_time, 2),
            "mean_distance_m": round(self.mean_distance, 1),
            "total_time_loss_s": round(self.total_time_loss, 1),
            "total_co2_g": round(self.total_co2_g, 1),
            "mean_network_speed_kmh": round(self.mean_network_speed, 2),
            "steps": self.steps,
            "seed": self.seed,
            "wall_time_s": round(self.wall_time, 2),
            "per_vehicle": self.per_vehicle,
            **self.meta,
        }


@dataclass(frozen=True, slots=True)
class SimulationTrace:
    """Where every vehicle was, sampled over the run.

    Positions rather than edge indices, because the point of a trace is to show
    the fleet *moving* — a plan animation replays this, and an index-based
    replay could only ever redraw the route it was already given.

    Grouped by vehicle rather than by frame.  A frame holds only the vehicles on
    the road at that instant, so a per-frame array would make the client
    re-derive which vehicle is which by nearest-neighbour matching — a guess
    that goes wrong the moment two vans pass on an adjacent lane.  Keying by
    identity keeps that certainty in the one place that has it, and lets a
    client interpolate a track instead of stepping it.
    """

    step_interval: float
    steps: int
    tracks: dict[str, tuple[tuple[float, float, float, float], ...]]

    def as_dict(self) -> dict:
        return {
            "step_interval": float(self.step_interval),
            "steps": int(self.steps),
            "tracks": {
                vid: [[round(t, 1), round(x, 1), round(y, 1), round(s, 2)] for t, x, y, s in points]
                for vid, points in self.tracks.items()
            },
        }


def build_trace(frames: Sequence[tuple[float, tuple[tuple[str, float, float, float], ...]]]) -> SimulationTrace:
    """Fold sampled frames into one track per vehicle."""
    tracks: dict[str, list[tuple[float, float, float, float]]] = {}
    for now, vehicles in frames:
        for vid, x, y, speed in vehicles:
            tracks.setdefault(vid, []).append((now, x, y, speed))
    interval = frames[1][0] - frames[0][0] if len(frames) > 1 else 1.0
    return SimulationTrace(
        step_interval=interval,
        steps=int(frames[-1][0]) if frames else 0,
        tracks={vid: tuple(points) for vid, points in tracks.items()},
    )


class FleetMonitor:
    """Per-vehicle bookkeeping across a whole run.

    SUMO's TraCI counters are per-step, not cumulative: ``getDepartedNumber``
    reports what left in the last step, and reading it once at the end of a run
    reports almost nothing.  A vehicle's own metrics are worse — they vanish
    with the vehicle, and there is no ``getTravelTime`` at all in 1.26 — so the
    monitor keeps the last snapshot of every vehicle and stamps the moment it
    disappeared.  That is the only way to recover a finished vehicle's journey
    time, and a report that silently averaged over the vehicles still on the
    road would flatter whichever method happened to be slowest.
    """

    FIELDS = ("distance", "time_loss", "co2", "waiting", "speed")

    def __init__(self) -> None:
        self.records: dict[str, dict] = {}
        self.departed = 0
        self.arrived = 0
        self._previous: set[str] = set()

    def step(self, traci, now: float) -> None:
        self.departed += int(traci.simulation.getDepartedNumber())
        self.arrived += int(traci.simulation.getArrivedNumber())
        present = set(traci.vehicle.getIDList())
        for vid in self._previous - present:
            record = self.records.get(vid)
            if record is not None and not record["finished"]:
                record["arrived_at"] = now
                record["finished"] = True
        self._previous = present
        for vid in present:
            self._refresh(traci, vid, now)

    def _refresh(self, traci, vid: str, now: float) -> None:
        record = self.records.get(vid)
        if record is None:
            depart = _safe(traci.vehicle.getDeparture, vid)
            record = {
                "departed_at": float(depart) if depart is not None else now,
                "arrived_at": now,
                "finished": False,
            }
            record.update(dict.fromkeys(self.FIELDS, 0.0))
            self.records[vid] = record
        for field, getter in (
            ("distance", "getDistance"),
            ("time_loss", "getTimeLoss"),
            ("co2", "getCO2Emission"),
            ("waiting", "getWaitingTime"),
            ("speed", "getSpeed"),
        ):
            value = _safe(getattr(traci.vehicle, getter), vid)
            if value is not None:
                record[field] = float(value)

    def finish(self, now: float) -> None:
        """Close out the vehicles that were still running when time ran out."""
        for record in self.records.values():
            if not record["finished"]:
                record["arrived_at"] = now

    def journey_times(self) -> list[float]:
        return [
            r["arrived_at"] - r["departed_at"] for r in self.records.values() if r["finished"]
        ]

    def mean(self, field_name: str, finished_only: bool = False) -> float:
        values = [
            r[field_name]
            for r in self.records.values()
            if (r["finished"] or not finished_only)
        ]
        return float(np.mean(values)) if values else 0.0

    def total(self, field_name: str) -> float:
        return float(sum(r[field_name] for r in self.records.values()))


def _safe(read, vehicle_id: str):
    try:
        return read(vehicle_id)
    except Exception:  # noqa: BLE001 - a vanished vehicle is not an error
        return None


class SumoRunner:
    """A supervised SUMO session driven over TraCI."""

    def __init__(self, net_file: str | Path, seed: int = 42) -> None:
        self.net_file = str(net_file)
        self.seed = int(seed)
        self.traci = import_traci()
        self.sumo_version = sumo_version()
        self._errors = self.traci.exceptions.TraCIException

    def command(
        self,
        routes_file: str | Path,
        steps: int,
        begin: float = 0.0,
        tripinfo: str | Path | None = None,
        emissions: str | Path | None = None,
    ) -> list[str]:
        cmd = [
            sumo_binary(),
            "-n",
            self.net_file,
            "-r",
            str(routes_file),
            "--begin",
            f"{begin:.2f}",
            "--end",
            f"{begin + steps:.2f}",
            "--step-length",
            "1.0",
            "--no-warnings",
            "true",
            "--no-step-log",
            "true",
            "--seed",
            str(self.seed),
            "--lateral-resolution",
            "0.8",
        ]
        if tripinfo is not None:
            cmd += ["--tripinfo-output", str(tripinfo)]
        if emissions is not None:
            cmd += [
                "--emission-output",
                str(emissions),
                "--emission-output.step-scaled",
            ]
        return cmd

    def tripinfo(self, path: str | Path) -> list[dict[str, float]]:
        """Per-vehicle journey totals from a SUMO tripinfo file.

        The authoritative source for anything that must be an integral, and the
        reason the report is not assembled from TraCI getters.  ``getCO2Emission``
        and ``getWaitingTime`` are *rates* in TraCI — the emissions and waiting of
        the current step — so reading them at the end of a run yields one step's
        value, and two vehicles stopped at a signal report byte-identical
        emissions.  The tripinfo file carries the integrals, and a vehicle still
        on the road when time runs out is simply absent from it, which is
        exactly the distinction the report needs to draw.

        SUMO 1.26's tripinfo carries no ``distance`` and no ``CO2``, so distance
        comes from the live session and emissions from the emission output.
        """
        target = Path(path)
        if not target.is_file():
            return []
        rows: list[dict[str, float]] = []
        for element in ET.parse(target).getroot():

            def value(name: str) -> float:
                try:
                    return float(element.get(name, "0") or 0.0)
                except ValueError:
                    return 0.0

            depart, arrival = value("depart"), value("arrival")
            rows.append(
                {
                    "id": element.get("id", ""),
                    "duration": arrival - depart,
                    "waiting": value("waitingTime"),
                    "time_loss": value("timeLoss"),
                    "route_length": value("routeLength"),
                }
            )
        return rows

    def emissions(self, path: str | Path) -> dict[str, float]:
        """Total CO2 per vehicle, in grams, from SUMO's emission output.

        The per-step rows are step-scaled on the command line, so the total is
        a plain sum over the rows for that vehicle.
        """
        target = Path(path)
        if not target.is_file():
            return {}
        totals: dict[str, float] = {}
        for element in ET.parse(target).getroot().iter("vehicle"):
            vid = element.get("id")
            if vid is None:
                continue
            try:
                totals[vid] = totals.get(vid, 0.0) + float(element.get("CO2", "0") or 0.0)
            except ValueError:
                continue
        return {vid: mg / 1000.0 for vid, mg in totals.items()}

    def run(
        self,
        routes_file: str | Path,
        steps: int = 3600,
        sample_every: int = 30,
        begin: float = 0.0,
        trace: bool = False,
        trace_every: int = 5,
    ) -> SimulationReport:
        """Simulate a routes file and return what the network actually did."""
        traci = self.traci
        started = time.perf_counter()
        tripinfo_path = Path(str(routes_file) + ".tripinfo.xml")
        emissions_path = Path(str(routes_file) + ".emission.xml")
        for scratch in (tripinfo_path, emissions_path):
            scratch.parent.mkdir(parents=True, exist_ok=True)
            scratch.unlink(missing_ok=True)
        traci.start(
            self.command(routes_file, steps, begin, tripinfo=tripinfo_path, emissions=emissions_path)
        )
        monitor = FleetMonitor()
        speeds: list[float] = []
        frames: list[tuple[float, tuple[tuple[str, float, float, float], ...]]] = []
        try:
            for step in range(int(steps)):
                traci.simulationStep()
                now = traci.simulation.getTime()
                monitor.step(traci, now)
                if step % sample_every == 0:
                    sample = monitor.mean("speed")
                    if sample > 0.0:
                        speeds.append(sample)
                if trace and step % trace_every == 0:
                    frames.append((now, self._fleet_positions()))
            monitor.finish(traci.simulation.getTime())
        finally:
            traci.close()
        report = self._collect(
            monitor, speeds, steps, started, self.tripinfo(tripinfo_path), self.emissions(emissions_path)
        )
        if trace and frames:
            report.meta["trace"] = build_trace(frames).as_dict()
        return report

    def _merge_tripinfo(
        self,
        observed: dict[str, dict],
        rows: Sequence[dict[str, float]],
        co2: dict[str, float],
    ) -> dict[str, dict]:
        """Fold the SUMO totals in over what the live session saw.

        A vehicle present in both keeps its live record — including the fact
        that it had not finished — and gains the exact totals SUMO integrated
        for it.  A vehicle the live session never saw (inserted and finished
        between two samples) is added from its tripinfo row alone.
        """
        merged: dict[str, dict] = {vid: dict(record) for vid, record in observed.items()}
        for row in rows:
            record = merged.setdefault(row["id"], {"finished": True, "travel_time": 0.0})
            record.update(
                travel_time=round(row["duration"], 1),
                waiting=round(row["waiting"], 1),
                time_loss=round(row["time_loss"], 1),
                co2=round(co2.get(row["id"], 0.0), 1),
                finished=True,
            )
        for vid, record in merged.items():
            record.setdefault("co2", round(co2.get(vid, 0.0), 1))
        return merged

    def _fleet_positions(self) -> tuple[tuple[str, float, float, float], ...]:
        """Every vehicle's identity, position and speed, right now."""
        traci = self.traci
        observed: list[tuple[str, float, float, float]] = []
        for vid in traci.vehicle.getIDList():
            xy = _safe(traci.vehicle.getPosition, vid)
            speed = _safe(traci.vehicle.getSpeed, vid)
            if xy is not None and speed is not None:
                observed.append((vid, float(xy[0]), float(xy[1]), float(speed)))
        return tuple(observed)

    def read_all(self, getter: str) -> list[float | None]:
        """Apply one TraCI vehicle getter to every vehicle, entry by entry."""
        read = getattr(self.traci.vehicle, getter)
        return [self._try(read, v) for v in self.traci.vehicle.getIDList()]

    def _try(self, read, vehicle_id: str) -> float | None:
        """Read one vehicle, tolerating the ones that finish mid-sample.

        The ID list is a snapshot, not a reservation: a vehicle that arrives
        between the listing and the read raises, and one arrival must not abort
        the whole run.
        """
        try:
            return read(vehicle_id)
        except self._errors:
            return None

    def _collect(
        self,
        monitor: FleetMonitor,
        speeds: list[float],
        steps: int,
        started: float,
        rows: Sequence[dict[str, float]],
        co2: dict[str, float],
    ) -> SimulationReport:
        """Turn the run's outcome into a report.

        Aggregates over the vehicles that *finished*, because a journey cut off
        at the horizon is not a shorter journey — averaging one in would credit
        whichever method's routes happened to still be running.  Distance, time
        loss and emissions are totals over the whole fleet, since those are
        quantities the fleet actually consumed.
        """
        per_vehicle = self._merge_tripinfo(
            {
                vid: {
                    "travel_time": round(record["arrived_at"] - record["departed_at"], 1),
                    "finished": bool(record["finished"]),
                    "distance": round(record["distance"], 1),
                    "time_loss": round(record["time_loss"], 1),
                    "co2": round(co2.get(vid, 0.0), 1),
                    "waiting": round(record["waiting"], 1),
                }
                for vid, record in sorted(monitor.records.items())
            },
            rows,
            co2,
        )
        journeys = [r["travel_time"] for r in per_vehicle.values() if r.get("finished")]
        departures = monitor.departed or len(per_vehicle)
        return SimulationReport(
            network=Path(self.net_file).stem,
            vehicles=len(per_vehicle),
            departures=departures,
            arrivals=monitor.arrived,
            mean_travel_time=float(np.mean(journeys)) if journeys else 0.0,
            finished_vehicles=len(journeys),
            mean_waiting_time=monitor.mean("waiting"),
            mean_distance=monitor.mean("distance"),
            total_time_loss=monitor.total("time_loss"),
            total_co2_g=monitor.total("co2"),
            mean_network_speed=float(np.mean(speeds)) if speeds else 0.0,
            completion_rate=monitor.arrived / departures if departures else 0.0,
            wall_time=time.perf_counter() - started,
            steps=int(steps),
            seed=self.seed,
            per_vehicle=per_vehicle,
        )

    def edge_speeds(self) -> dict[str, float]:
        """Current mean speed on every edge, for the dynamic-routing experiments."""
        traci = self.traci
        return dict(
            zip(traci.edge.getIDList(), traci.edge.getLastStepMeanSpeed(), strict=True)
        )

    def with_session(self, routes_file: str | Path, steps: int, begin: float = 0.0):
        """A context manager owning a live connection, for interactive use."""
        return _Session(self, routes_file, steps, begin)


class _Session:
    """Keeps a TraCI connection open across several steps."""

    def __init__(self, runner: SumoRunner, routes_file: str | Path, steps: int, begin: float) -> None:
        self.runner = runner
        self.steps = int(steps)
        self.begin = float(begin)
        self.routes_file = routes_file

    def __enter__(self):
        self.runner.traci.start(self.runner.command(self.routes_file, self.steps, self.begin))
        return self.runner.traci

    def __exit__(self, *exc) -> None:
        self.runner.traci.close()


def expand_solution(
    network: RoadNetwork,
    routes: Sequence[Sequence[int]],
    depot: int,
    customer_nodes: Sequence[int],
    edge_cost: np.ndarray,
) -> tuple[
    list[tuple[int, ...]],
    dict[int, tuple[int, ...]],
    list[int],
    list[list[tuple[int, int]]],
]:
    """Turn a decoded solution into one drivable edge path per vehicle.

    Each route is planned as a *chain*, not as independent trips between
    consecutive stops.  Every leg after the first is planned with
    :func:`dijkstra_continuing`, which may only leave the previous leg through
    a legal successor of the edge the vehicle just drove.  Planning the legs
    independently and concatenating them looks equivalent and is not: two legal
    paths can meet at a junction with no permitted continuation between them,
    and the joined route is then unbuildable at load time, which would silently
    invalidate every number measured from the run.

    Returns the per-vehicle edge paths, the position within each path where that
    vehicle's deliveries happen, the customers no vehicle could reach, and — per
    vehicle — the ``(position, customer)`` pairs in visit order.  The last of
    these is what a caller needs to say *which* stop a vehicle is heading for;
    the service positions alone say where, and not what, and a consumer that
    guessed the identity by matching coordinates would be guessing.
    """
    legs: list[tuple[int, ...]] = []
    stops: dict[int, tuple[int, ...]] = {}
    unserved: list[int] = []
    deliveries: list[list[tuple[int, int]]] = []

    for route in routes:
        customers = [int(c) for c in route]
        if not customers:
            continue
        targets = [int(customer_nodes[c]) for c in customers]
        edges: list[int] = []
        positions: list[int] = []
        reached: list[tuple[int, int]] = []
        here = depot
        arrival = -1

        for customer, target in enumerate(targets):
            if arrival >= 0 and network.edges[arrival].to_node == target:
                positions.append(len(edges) - 1)
                reached.append((len(edges) - 1, customers[customer]))
                continue
            leg = (
                dijkstra(network, edge_cost, depot, target)
                if arrival < 0
                else dijkstra_continuing(network, edge_cost, arrival, target)
            )
            if not leg.edges and target != depot:
                unserved.append(customers[customer])
                continue
            edges.extend(leg.edges)
            if edges:
                arrival = int(edges[-1])
                here = target
                positions.append(len(edges) - 1)
                reached.append((len(edges) - 1, customers[customer]))

        if arrival >= 0 and here != depot:
            back = dijkstra_continuing(network, edge_cost, arrival, depot)
            edges.extend(back.edges)

        if edges:
            stops[len(legs)] = tuple(positions)
            legs.append(tuple(edges))
            deliveries.append(reached)

    return legs, stops, unserved, deliveries
