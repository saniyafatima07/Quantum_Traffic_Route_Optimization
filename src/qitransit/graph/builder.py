"""Synthetic urban network generator and the bridge to ``netconvert``.

Two directions are supported:

* :func:`build_city` lays out a road graph in Python and emits SUMO's *plain*
  XML (``.nod`` / ``.edg`` / ``.con`` / ``.tllogic``), which ``netconvert``
  compiles into a ``.net.xml``.  Having the abstract graph available before
  simulation is what lets the optimisers work on the same network object the
  simulator uses.
* :func:`import_network` hands real-world OpenStreetMap extracts to
  ``netconvert`` and reads the compiled network back into the same model.
"""

from __future__ import annotations

import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import sumo_env
from .network import Edge, Node, RoadNetwork

ROAD_CLASSES = {
    "arterial": {"lanes": 2, "speed": 13.89, "priority": 3},
    "collector": {"lanes": 2, "speed": 11.11, "priority": 2},
    "local": {"lanes": 1, "speed": 8.33, "priority": 1},
}

TLS_CYCLE_GREEN = 31.0
TLS_YELLOW = 4.0


@dataclass(frozen=True, slots=True)
class CitySpec:
    """Parameters of a synthetic city layout."""

    name: str = "downtown"
    rows: int = 7
    cols: int = 7
    block_length: float = 220.0
    arterial_every: int = 3
    jitter: float = 0.12
    signal_offset: float = 0.0
    seed: int = 20260101

    @property
    def label(self) -> str:
        return f"{self.name}_{self.rows}x{self.cols}_s{self.seed}"


PRESETS: dict[str, CitySpec] = {
    "micro": CitySpec("micro", rows=4, cols=4, arterial_every=2),
    "manhattan": CitySpec("manhattan", rows=8, cols=8, arterial_every=3),
    "downtown": CitySpec("downtown", rows=12, cols=12, arterial_every=4),
    "metro": CitySpec("metro", rows=18, cols=18, arterial_every=5),
    "megacity": CitySpec("megacity", rows=26, cols=26, arterial_every=6),
}


def preset(name: str, **overrides) -> CitySpec:
    """Look up a named preset, optionally overriding individual fields."""
    if name not in PRESETS:
        raise KeyError(f"unknown city preset {name!r}; choose from {sorted(PRESETS)}")
    return replace(PRESETS[name], **overrides)


# ----------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------
def _classify(row: int, col: int, rows: int, cols: int, arterial_every: int) -> str:
    on_arterial_row = arterial_every > 0 and (row % arterial_every == 0 or row == rows)
    on_arterial_col = arterial_every > 0 and (col % arterial_every == 0 or col == cols)
    if on_arterial_row or on_arterial_col:
        return "arterial"
    if (row + col) % 2 == 0:
        return "collector"
    return "local"


def build_city(spec: CitySpec) -> RoadNetwork:
    """Lay out a street grid with a road-class hierarchy and jittered blocks."""
    rng = np.random.default_rng(spec.seed)
    rows, cols = spec.rows, spec.cols

    xs = [0.0]
    for c in range(1, cols + 1):
        step = spec.block_length * (1.0 + rng.uniform(-spec.jitter, spec.jitter))
        xs.append(xs[-1] + step)
    ys = [0.0]
    for r in range(1, rows + 1):
        step = spec.block_length * (1.0 + rng.uniform(-spec.jitter, spec.jitter))
        ys.append(ys[-1] + step)

    index_of: dict[tuple[int, int], int] = {}
    nodes: list[Node] = []
    for r in range(rows + 1):
        for c in range(cols + 1):
            if 0 < r < rows and 0 < c < cols:
                jitter_x = float(rng.uniform(-8.0, 8.0))
                jitter_y = float(rng.uniform(-8.0, 8.0))
            else:
                jitter_x = jitter_y = 0.0
            idx = len(nodes)
            index_of[(r, c)] = idx
            nodes.append(
                Node(
                    id=f"J{r}_{c}",
                    index=idx,
                    x=xs[c] + jitter_x,
                    y=ys[r] + jitter_y,
                    has_traffic_light=False,
                )
            )

    edges: list[Edge] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def add_arc(a: tuple[int, int], b: tuple[int, int], road_class: str) -> None:
        if (a, b) in seen:
            return
        seen.add((a, b))
        profile = ROAD_CLASSES[road_class]
        tail, head = index_of[a], index_of[b]
        dx = nodes[head].x - nodes[tail].x
        dy = nodes[head].y - nodes[tail].y
        length = float(np.hypot(dx, dy))
        eid = f"E{len(edges)}"
        edges.append(
            Edge(
                id=eid,
                index=len(edges),
                from_node=tail,
                to_node=head,
                length=length,
                speed=float(profile["speed"]) * (1.0 + float(rng.uniform(-0.05, 0.05))),
                lanes=int(profile["lanes"]),
                priority=int(profile["priority"]),
                road_class=road_class,
            )
        )

    for r in range(rows + 1):
        for c in range(cols + 1):
            here = (r, c)
            road_class = _classify(r, c, rows, cols, spec.arterial_every)
            if c < cols:
                nxt = _classify(r, c + 1, rows, cols, spec.arterial_every)
                add_arc(here, (r, c + 1), max(road_class, nxt, key=_rank))
                add_arc((r, c + 1), here, max(road_class, nxt, key=_rank))
            if r < rows:
                nxt = _classify(r + 1, c, rows, cols, spec.arterial_every)
                add_arc(here, (r + 1, c), max(road_class, nxt, key=_rank))
                add_arc((r + 1, c), here, max(road_class, nxt, key=_rank))

    network = RoadNetwork(nodes, edges)
    signal_nodes = _pick_signal_nodes(network, index_of, rows, cols, spec.arterial_every)
    if signal_nodes:
        nodes = [
            Node(n.id, n.index, n.x, n.y, has_traffic_light=(n.index in signal_nodes))
            for n in nodes
        ]
        network = RoadNetwork(nodes, edges)
    return network


def _rank(road_class: str) -> int:
    return {"local": 1, "collector": 2, "arterial": 3}[road_class]


def _pick_signal_nodes(
    network: RoadNetwork, index_of: dict, rows: int, cols: int, arterial_every: int
) -> set[int]:
    """Signalise junctions where at least one arterial leg meets another road."""
    chosen: set[int] = set()
    for r in range(1, rows):
        for c in range(1, cols):
            if arterial_every <= 0:
                continue
            if r % arterial_every == 0 or c % arterial_every == 0:
                chosen.add(index_of[(r, c)])
    if not chosen:
        chosen = {index_of[(r, c)] for r in range(1, rows) for c in range(1, cols)}
    return chosen


# ----------------------------------------------------------------------
# plain XML emission
# ----------------------------------------------------------------------
def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def write_plain_xml(network: RoadNetwork, out_dir: str | Path) -> dict[str, Path]:
    """Emit ``.nod`` / ``.edg`` / ``.con`` for ``netconvert``.

    The traffic-light program is written separately by :func:`write_tl_logic`
    because its phase count must match the number of links ``netconvert`` ends
    up building at each junction.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    nod = out / "city.nod.xml"
    with nod.open("w", encoding="utf-8") as fh:
        fh.write("<nodes>\n")
        for node in network.nodes:
            kind = "traffic_light" if node.has_traffic_light else "priority"
            fh.write(
                f'    <node id="{_escape(node.id)}" x="{node.x:.2f}" y="{node.y:.2f}" '
                f'type="{kind}"/>\n'
            )
        fh.write("</nodes>\n")

    edg = out / "city.edg.xml"
    with edg.open("w", encoding="utf-8") as fh:
        fh.write("<edges>\n")
        for edge in network.edges:
            fh.write(
                f'    <edge id="{_escape(edge.id)}" from="{_escape(network.nodes[edge.from_node].id)}" '
                f'to="{_escape(network.nodes[edge.to_node].id)}" numLanes="{edge.lanes}" '
                f'speed="{edge.speed:.2f}" priority="{edge.priority}"/>\n'
            )
        fh.write("</edges>\n")

    con = out / "city.con.xml"
    with con.open("w", encoding="utf-8") as fh:
        fh.write("<connections>\n")
        for edge in network.edges:
            head = edge.to_node
            for target in network.out_edges(head):
                other = network.edges[int(target)]
                if other.from_node != edge.to_node:
                    continue
                if other.index == network.edge_between(head, edge.from_node):
                    continue
                if not network.nodes[head].has_traffic_light and other.priority < edge.priority:
                    continue
                fh.write(
                    f'    <connection from="{_escape(edge.id)}" to="{_escape(other.id)}" '
                    f'fromLane="0" toLane="0"/>\n'
                )
        fh.write("</connections>\n")

    return {"nodes": nod, "edges": edg, "connections": con}


def write_tl_logic(
    network: RoadNetwork,
    path: str | Path,
    link_groups: dict[str, list[list[int]]] | None = None,
) -> Path:
    """Write a fixed-time signal program for every signalised junction.

    ``link_groups`` maps a junction id to its approaches, each of which is the
    list of link indices belonging to one incoming road.  Those groups are read
    back from ``netconvert``'s own output because a multi-lane approach owns
    several link indices; cycling whole approaches (rather than individual
    link indices) is what makes the signal behave like a real intersection.
    When the mapping is omitted the abstract in-edge count is used.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        fh.write("<tlLogics>\n")
        for node in network.nodes:
            if not node.has_traffic_light:
                continue
            groups = (link_groups or {}).get(node.id)
            if not groups:
                groups = [[i] for i in range(max(2, len(network.in_edges(node.index))))]
            n_links = sum(len(g) for g in groups)
            fh.write(
                f'    <tlLogic id="{_escape(node.id)}" type="static" programID="0" offset="0">\n'
            )
            for k, group in enumerate(groups):
                green = ["r"] * n_links
                for link in group:
                    green[link] = "G"
                fh.write(
                    f'        <phase duration="{TLS_CYCLE_GREEN:.0f}" state="{"".join(green)}"/>\n'
                )
                amber = ["r"] * n_links
                for link in group:
                    amber[link] = "y"
                fh.write(
                    f'        <phase duration="{TLS_YELLOW:.0f}" state="{"".join(amber)}"/>\n'
                )
            fh.write("    </tlLogic>\n")
        fh.write("</tlLogics>\n")
    return target


def _tls_link_groups(net_file: Path) -> dict[str, list[list[int]]]:
    """Group each signalised junction's link indices by incoming road.

    ``netconvert`` splits a wide approach into one link index per lane group,
    and it may split roads at junctions, so the grouping is read back from the
    compiled network rather than assumed from the abstract graph.
    """
    root = ET.parse(net_file).getroot()
    groups: dict[str, list[list[int]]] = {}
    for logic in root.iter("tlLogic"):
        node_id = logic.get("id")
        by_approach: dict[str, list[int]] = {}
        for conn in logic.iter("connection"):
            link = conn.get("linkIndex")
            source = conn.get("from")
            if link is None or source is None:
                continue
            by_approach.setdefault(source, []).append(int(link))
        ordered = [sorted(v) for _, v in sorted(by_approach.items())]
        if not ordered:
            continue
        groups[node_id] = ordered
    return groups


BASE_NETCONVERT_FLAGS = (
    "--no-turnarounds",
    "--no-turnarounds.tls",
    "--junctions.limit-turn-speed",
    "5",
    "--default.junctions.keep-clear",
    "--tls.default-type",
    "static",
    "--no-warnings",
    "-X",
    "never",
)

NETCONVERT_FLAGS = BASE_NETCONVERT_FLAGS


def _run_netconvert(cmd: list[str], target: Path) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not target.exists():
        raise RuntimeError(
            f"netconvert failed ({result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}\ncommand: {' '.join(cmd)}"
        )


def compile_network(
    network: RoadNetwork,
    out_dir: str | Path,
    name: str = "city",
    fixed_time_signals: bool = True,
    max_tls_passes: int = 5,
) -> Path:
    """Write plain XML, run ``netconvert`` and return the ``.net.xml`` path.

    With ``fixed_time_signals`` the traffic-light program is written
    explicitly.  How many link indices a junction owns depends on whether the
    program is supplied or guessed, so the program is rewritten and recompiled
    until the junction layout stops changing, which takes two or three passes.
    """
    out = Path(out_dir)
    files = write_plain_xml(network, out)
    target = out / f"{name}.net.xml"
    tll_path = out / "city.tllogic.xml"

    base = [
        sumo_env.netconvert_binary(),
        "-n",
        str(files["nodes"]),
        "-e",
        str(files["edges"]),
        "-x",
        str(files["connections"]),
        "-o",
        str(target),
    ]

    if not fixed_time_signals:
        _run_netconvert([*base, "--tls.guess", *BASE_NETCONVERT_FLAGS], target)
        return target

    _run_netconvert([*base, "--tls.guess", *BASE_NETCONVERT_FLAGS], target)
    previous: str | None = None
    for _ in range(max_tls_passes):
        write_tl_logic(network, tll_path, _tls_link_groups(target))
        current = tll_path.read_text(encoding="utf-8")
        if current == previous:
            break
        previous = current
        _run_netconvert([*base, "-i", str(tll_path), *BASE_NETCONVERT_FLAGS], target)
    else:  # pragma: no cover - only on a pathological junction layout
        raise RuntimeError(f"traffic-light program did not converge for {name}")
    return target


# ----------------------------------------------------------------------
# reading compiled / real networks
# ----------------------------------------------------------------------
def import_network(
    source: str | Path,
    out_dir: str | Path,
    name: str = "imported",
    osm: bool = False,
) -> tuple[RoadNetwork, Path]:
    """Compile a network from OSM/plain files and load it back into the model."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{name}.net.xml"

    if target.exists() and target.stat().st_mtime >= Path(source).stat().st_mtime:
        return load_net_file(target), target

    source_path = Path(source)
    if osm:
        cmd = [sumo_env.netconvert_binary(), "--osm-files", str(source_path), "-o", str(target)]
    elif source_path.suffix == ".xml":
        cmd = [sumo_env.netconvert_binary(), "-s", str(source_path), "-o", str(target)]
    else:
        raise ValueError(
            "import_network expects an .osm/.xml file; use build_city() for plain files"
        )
    cmd.extend(NETCONVERT_FLAGS)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not target.exists():
        raise RuntimeError(f"netconvert failed:\n{result.stdout}\n{result.stderr}")
    return load_net_file(target), target


def load_net_file(
    path: str | Path, vehicle_class: str = "delivery"
) -> RoadNetwork:
    """Read a compiled ``.net.xml`` into the in-memory :class:`RoadNetwork`.

    WGS84 positions are recovered from SUMO's projection when the file carries
    one, so customers can later be placed at real addresses.

    The legal turns are read as well, filtered to ``vehicle_class``.  Without
    them the network would be a relaxed abstraction, and routes found on it
    would not necessarily be drivable: ``netconvert`` runs here disable
    U-turns, and OSM carries per-lane access restrictions that the node
    adjacency alone cannot express.

    Successors naming an edge outside the loaded edge set are dropped rather
    than raising.  On a well-formed net this is a no-op; it is defensive
    because a network where it does fire would otherwise be unloadable, and
    because dropping is the conservative direction — the turn graph then offers
    fewer continuations than the simulator would, so any route found here stays
    drivable there.
    """
    sumolib = sumo_env.import_sumolib()
    net = sumolib.net.readNet(str(path), withPrograms=True)

    xs: list[float] = []
    ys: list[float] = []
    types: list[str] = []
    ids: list[str] = []
    for node in net.getNodes():
        x, y = node.getCoord()
        ids.append(node.getID())
        xs.append(float(x))
        ys.append(float(y))
        types.append(str(node.getType()))

    lonlat = np.zeros((len(ids), 2), dtype=np.float64)
    if net.hasGeoProj():
        lonlat = np.array([net.convertXY2LonLat(x, y) for x, y in zip(xs, ys, strict=True)])

    nodes = [
        Node(
            id=node_id,
            index=i,
            x=xs[i],
            y=ys[i],
            has_traffic_light=types[i] == "traffic_light",
            lat=float(lonlat[i, 1]) or None,
            lon=float(lonlat[i, 0]) or None,
        )
        for i, node_id in enumerate(ids)
    ]

    index_of_node = {node_id: i for i, node_id in enumerate(ids)}
    sumo_edges = list(net.getEdges())
    index_of_edge = {edge.getID(): i for i, edge in enumerate(sumo_edges)}
    edges: list[Edge] = []
    for i, edge in enumerate(sumo_edges):
        try:
            road_class = str(edge.getType())
        except Exception:  # pragma: no cover - degenerate edge attributes
            road_class = "urban"
        try:
            priority = int(edge.getPriority())
        except Exception:  # pragma: no cover
            priority = 1
        edges.append(
            Edge(
                id=edge.getID(),
                index=i,
                from_node=index_of_node[edge.getFromNode().getID()],
                to_node=index_of_node[edge.getToNode().getID()],
                length=float(edge.getLength()),
                speed=float(edge.getSpeed()),
                lanes=int(edge.getLaneNumber()),
                priority=priority,
                road_class=road_class,
            )
        )

    connections = [
        [
            index_of_edge[nxt.getID()]
            for nxt in edge.getAllowedOutgoing(vehicle_class)
            if nxt.getID() in index_of_edge
        ]
        for edge in sumo_edges
    ]
    return RoadNetwork(nodes, edges, connections)


def build_preset(name: str, out_dir: str | Path, **overrides) -> tuple[RoadNetwork, Path]:
    """Convenience wrapper: preset -> graph -> compiled ``.net.xml``."""
    spec = preset(name, **overrides)
    network = build_city(spec)
    target = compile_network(network, out_dir, name=spec.label)
    return network, target


def network_from_sumo(path: str | Path) -> RoadNetwork:
    return load_net_file(path)
