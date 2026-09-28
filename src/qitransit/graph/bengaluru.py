"""The Bengaluru road network, built from real OpenStreetMap data.

Bengaluru is the study area.  The module downloads a Geofabrik extract of
southern India (Karnataka included) once, clips it to a bounding box with
``osmium`` in three streaming passes, and hands the result to ``netconvert`` so
that the optimiser and the simulator operate on the same graph.

Landmark coordinates for the named districts are used to describe the
extraction area and to place depots and customers in realistic locations.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .builder import load_net_file
from .network import RoadNetwork

GEOFABRIK_URL = (
    "https://download.geofabrik.de/asia/india/southern-zone-latest.osm.pbf"
)
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"

DRIVABLE_HIGHWAYS = frozenset(
    {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "unclassified",
        "residential",
        "living_street",
        "road",
        "service",
    }
)

INTERESTING_NODE_TAGS = ("highway", "junction", "traffic_signals", "barrier", "crossing", "public_transport")

INTERESTING_NODE_VALUES = frozenset(
    {
        "traffic_signals",
        "traffic_light",
        "stop",
        "give_way",
        "crossing",
        "turning_circle",
        "motorway_junction",
        "junction",
        "traffic_calming",
        "bump",
        "block",
        "gate",
        "lift_gate",
        "cycle_barrier",
        "bus_stop",
    }
)


@dataclass(frozen=True, slots=True)
class BBox:
    """A geographic window, ``south``-``west``-``north``-``east``."""

    south: float
    west: float
    north: float
    east: float

    def contains(self, lat: float, lon: float) -> bool:
        return self.south <= lat <= self.north and self.west <= lon <= self.east

    def expanded(self, margin: float) -> BBox:
        return BBox(
            self.south - margin,
            self.west - margin,
            self.north + margin,
            self.east + margin,
        )

    def as_netconvert(self) -> str:
        return f"{self.west},{self.south},{self.east},{self.north}"

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.south, self.west, self.north, self.east)

    @property
    def size_km(self) -> tuple[float, float]:
        mid_lat = (self.south + self.north) / 2.0
        height = (self.north - self.south) * 110.574
        width = (self.east - self.west) * 111.320 * float(np.cos(np.radians(mid_lat)))
        return (round(height, 2), round(width, 2))


AREAS: dict[str, BBox] = {
    "central": BBox(12.9050, 77.5350, 12.9950, 77.7050),
    "inner_ring": BBox(12.8300, 77.4600, 13.0600, 77.7800),
    "metropolitan": BBox(12.7200, 77.3500, 13.2500, 77.9000),
    "bengaluru": BBox(12.5500, 77.2000, 13.4500, 78.0500),
}

LANDMARKS: dict[str, tuple[float, float]] = {
    "Cubbon Park": (12.9763, 77.6033),
    "MG Road": (12.9750, 77.6068),
    "Brigade Road": (12.9716, 77.6070),
    "Vidhana Soudha": (12.9794, 77.5912),
    "UB City": (12.9719, 77.5946),
    "Shivajinagar": (12.9850, 77.6050),
    "Seshadripuram": (12.9917, 77.5773),
    "Malleshwaram": (13.0035, 77.5700),
    "Rajajinagar": (12.9915, 77.5520),
    "Yeshwanthpur": (13.0234, 77.5540),
    "Hebbal": (13.0358, 77.5970),
    "Yelahanka": (13.1007, 77.5963),
    "Indiranagar": (12.9719, 77.6412),
    "Domlur": (12.9611, 77.6387),
    "CV Raman Nagar": (12.9917, 77.6360),
    "Koramangala": (12.9352, 77.6245),
    "HSR Layout": (12.9116, 77.6474),
    "BTM Layout": (12.9166, 77.6101),
    "Silk Board": (12.9172, 77.6226),
    "Jayanagar": (12.9250, 77.5938),
    "Banashankari": (12.9250, 77.5667),
    "Basavanagudi": (12.9425, 77.5730),
    "Lalbagh": (12.9507, 77.5848),
    "Kengeri": (12.9081, 77.4850),
    "Peenya": (13.0290, 77.5190),
    "Whitefield": (12.9698, 77.7499),
    "Hope Farm": (12.9870, 77.7500),
    "Marathahalli": (12.9591, 77.6974),
    "Bellandur": (12.9304, 77.6784),
    "Sarjapur Road": (12.9081, 77.6851),
    "KR Puram": (12.9694, 77.6947),
    "Madiwala": (12.9750, 77.7000),
    "Electronic City": (12.8452, 77.6602),
    "Bommasandra": (12.9008, 77.6146),
    "Manyata Tech Park": (13.0475, 77.6206),
    "Kempegowda Airport": (13.1986, 77.7066),
    "Devanahalli": (13.2437, 77.7128),
    "Nelamangala": (13.0990, 77.3920),
    "Doddaballapur": (13.1750, 77.5280),
    "Nelamangala Road": (13.0800, 77.4500),
}

LANDMARKS_BY_AREA: dict[str, list[str]] = {
    "central": [
        "Jayanagar",
        "Basavanagudi",
        "Banashankari",
        "Lalbagh",
        "Cubbon Park",
        "Vidhana Soudha",
        "MG Road",
        "UB City",
        "Shivajinagar",
        "Koramangala",
        "BTM Layout",
        "HSR Layout",
        "Silk Board",
        "Indiranagar",
        "Domlur",
        "CV Raman Nagar",
    ],
    "inner_ring": [
        "Kengeri",
        "Electronic City",
        "Bommasandra",
        "BTM Layout",
        "HSR Layout",
        "Silk Board",
        "Koramangala",
        "Sarjapur Road",
        "Bellandur",
        "Indiranagar",
        "Domlur",
        "Whitefield",
        "Hope Farm",
        "Marathahalli",
        "KR Puram",
        "Madiwala",
        "Peenya",
        "Hebbal",
        "Yeshwanthpur",
        "Malleshwaram",
        "Rajajinagar",
        "Manyata Tech Park",
        "Yelahanka",
    ],
    "metropolitan": [
        "Kengeri",
        "Electronic City",
        "Bommasandra",
        "HSR Layout",
        "Koramangala",
        "Sarjapur Road",
        "Whitefield",
        "Marathahalli",
        "Indiranagar",
        "MG Road",
        "Vidhana Soudha",
        "Peenya",
        "Hebbal",
        "Yeshwanthpur",
        "Manyata Tech Park",
        "Yelahanka",
        "Nelamangala",
        "Nelamangala Road",
        "Doddaballapur",
        "Kempegowda Airport",
        "Devanahalli",
    ],
    "bengaluru": [
        "Kengeri",
        "Electronic City",
        "Whitefield",
        "MG Road",
        "Peenya",
        "Hebbal",
        "Yelahanka",
        "Nelamangala",
        "Doddaballapur",
        "Kempegowda Airport",
        "Devanahalli",
    ],
}


def area(name: str) -> BBox:
    if name not in AREAS:
        raise KeyError(f"unknown Bengaluru area {name!r}; choose from {sorted(AREAS)}")
    return AREAS[name]


# ----------------------------------------------------------------------
# source data
# ----------------------------------------------------------------------
def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "osm"


def source_pbf(data_dir: str | Path | None = None, allow_download: bool = True) -> Path:
    """Path of the Geofabrik extract, downloading it on first use."""
    root = Path(data_dir) if data_dir else default_data_dir()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "southern-zone.osm.pbf"
    if target.exists() and target.stat().st_size > 100_000_000:
        return target
    if not allow_download:
        raise FileNotFoundError(
            f"{target} is missing. Run `qitransit bengaluru fetch` with network access."
        )
    fetch(target)
    return target


def fetch(target: str | Path, url: str = GEOFABRIK_URL, force: bool = False) -> Path:
    """Download the Geofabrik southern-zone extract.

    A browser ``User-Agent`` is required: several networks in front of the
    public mirrors reject programmatic agents outright.
    """
    target = Path(target)
    if target.exists() and not force and target.stat().st_size > 100_000_000:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
    started = time.time()
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as fh:
        shutil.copyfileobj(response, fh, length=1 << 22)
    size_mb = target.stat().st_size / 1e6
    print(
        f"downloaded {size_mb:.0f} MB from {url} in {time.time() - started:.0f}s -> {target}"
    )
    if size_mb < 50:
        raise RuntimeError(f"download looks truncated ({size_mb:.1f} MB)")
    return target


# ----------------------------------------------------------------------
# clipping
# ----------------------------------------------------------------------
def _relevant_node(node) -> bool:
    if not node.tags:
        return False
    for key in INTERESTING_NODE_TAGS:
        value = node.tags.get(key)
        if value is None:
            continue
        if key in {"highway", "junction", "barrier", "crossing"} and value in INTERESTING_NODE_VALUES:
            return True
        if key == "traffic_signals":
            return True
    return False


def _way_is_drivable(way) -> bool:
    if not way.tags:
        return False
    highway = way.tags.get("highway")
    if highway in DRIVABLE_HIGHWAYS:
        return True
    return bool(way.tags.get("route") == "road") and highway is not None
def _xml_attr(value) -> str:
    """Escape a value for use inside a single-quoted XML attribute."""
    text = str(value)
    for src, dst in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ("'", "&apos;"), ('"', "&quot;")):
        text = text.replace(src, dst)
    return text


def extract_osm(
    pbf: str | Path,
    bbox: BBox,
    out_dir: str | Path,
    name: str = "bengaluru",
    progress: bool = True,
    margin: float = 0.004,
) -> Path:
    """Clip ``pbf`` to ``bbox`` and write a plain OSM XML file.

    Four streaming passes keep memory proportional to the clip rather than to
    the extract, and none of them build a node location index:

    1. nodes inside the padded box are recorded, and those inside the core box
       are flagged as "in scope";
    2. drivable ways touching an in-scope node are selected, together with every
       node id they reference;
    3. the selected nodes are written, plus tagged junctions in the padded box
       (traffic signals drive where SUMO places signals);
    4. the selected ways and the turn restrictions that apply to them are
       written, resolving geometry from the coordinates kept in pass 1.
    """
    import osmium

    pbf = Path(pbf)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{name}.osm"
    if target.exists() and target.stat().st_size > 0:
        return target

    t0 = time.time()
    padded = bbox.expanded(margin)

    coords: dict[int, tuple[float, float]] = {}
    in_scope: set[int] = set()

    class NodeIndex(osmium.SimpleHandler):
        def node(self, node) -> None:
            lat, lon = node.location.lat, node.location.lon
            if not padded.contains(lat, lon):
                return
            coords[node.id] = (lat, lon)
            if bbox.contains(lat, lon):
                in_scope.add(node.id)

    NodeIndex().apply_file(str(pbf))
    if progress:
        print(
            f"  pass 1: {len(coords)} nodes indexed, {len(in_scope)} in scope "
            f"({time.time() - t0:.0f}s)"
        )

    way_ids: set[int] = set()
    needed: set[int] = set()

    class WayScan(osmium.SimpleHandler):
        def way(self, way) -> None:
            if not _way_is_drivable(way):
                return
            refs = [n.ref for n in way.nodes]
            if not any(r in in_scope for r in refs):
                return
            way_ids.add(way.id)
            needed.update(refs)

    WayScan().apply_file(str(pbf))
    if progress:
        print(
            f"  pass 2: {len(way_ids)} drivable ways, {len(needed)} referenced nodes "
            f"({time.time() - t0:.0f}s)"
        )

    written: set[int] = set()
    fh = target.open("w", encoding="utf-8")
    fh.write("<?xml version=\'1.0\' encoding=\'UTF-8\'?>\n")
    fh.write("<osm version=\'0.6\' generator=\'qitransit\'>\n")

    def emit_node(node_id: int, lat: float, lon: float, tags) -> None:
        if not tags:
            fh.write(f"  <node id=\'{node_id}\' lat=\'{lat:.7f}\' lon=\'{lon:.7f}\'/>\n")
            return
        fh.write(f"  <node id=\'{node_id}\' lat=\'{lat:.7f}\' lon=\'{lon:.7f}\'>\n")
        for tag in tags:
            fh.write(
                f"    <tag k=\'{_xml_attr(tag.k)}\' v=\'{_xml_attr(tag.v)}\'/>\n"
            )
        fh.write("  </node>\n")

    class NodeWrite(osmium.SimpleHandler):
        def node(self, node) -> None:
            lat, lon = node.location.lat, node.location.lon
            if node.id in needed:
                tags = list(node.tags)
            elif padded.contains(lat, lon) and _relevant_node(node):
                tags = [t for t in node.tags if t.k in INTERESTING_NODE_TAGS]
            else:
                return
            emit_node(node.id, lat, lon, tags)
            written.add(node.id)

    NodeWrite().apply_file(str(pbf))
    if progress:
        print(f"  pass 3: {len(written)} nodes written ({time.time() - t0:.0f}s)")

    missing_geometry = 0
    ways_written = 0

    class WayWrite(osmium.SimpleHandler):
        def way(self, way) -> None:
            nonlocal missing_geometry, ways_written
            if way.id not in way_ids:
                return
            refs = [n.ref for n in way.nodes]
            resolved = [r for r in refs if r in coords]
            if not resolved:
                return
            lats = np.fromiter((coords[r][0] for r in resolved), dtype=np.float64, count=len(resolved))
            lons = np.fromiter((coords[r][1] for r in resolved), dtype=np.float64, count=len(resolved))
            if not (
                bbox.north >= lats.min()
                and lats.max() >= bbox.south
                and bbox.east >= lons.min()
                and lons.max() >= bbox.west
            ):
                return
            for ref in refs:
                if ref in written or ref not in coords:
                    continue
                lat, lon = coords[ref]
                emit_node(ref, lat, lon, ())
                written.add(ref)
                missing_geometry += 1
            fh.write(f"  <way id=\'{way.id}\'>\n")
            for ref in refs:
                fh.write(f"    <nd ref=\'{ref}\'/>\n")
            for tag in way.tags:
                fh.write(f"    <tag k=\'{_xml_attr(tag.k)}\' v=\'{_xml_attr(tag.v)}\'/>\n")
            fh.write("  </way>\n")
            ways_written += 1

        def relation(self, relation) -> None:
            rtype = relation.tags.get("type")
            if rtype not in {"restriction", "restriction:bus", "restriction:car"}:
                return
            members = list(relation.members)
            if not any(m.type == "w" and m.ref in way_ids for m in members):
                return
            fh.write(f"  <relation id=\'{relation.id}\'>\n")
            for member in members:
                fh.write(
                    f"    <member type=\'{member.type}\' ref=\'{member.ref}\' "
                    f"role=\'{_xml_attr(member.role or "")}\'/>\n"
                )
            for tag in relation.tags:
                fh.write(f"    <tag k=\'{_xml_attr(tag.k)}\' v=\'{_xml_attr(tag.v)}\'/>\n")
            fh.write("  </relation>\n")

    WayWrite().apply_file(str(pbf))
    fh.write("</osm>\n")
    fh.close()

    if progress:
        print(
            f"  pass 4: {ways_written} ways, {len(written)} nodes "
            f"({missing_geometry} boundary nodes) in {time.time() - t0:.0f}s"
        )
        print(f"  wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


OSM_NETCONVERT_FLAGS = (
    "--no-turnarounds",
    "--junctions.limit-turn-speed",
    "5",
    "--default.junctions.keep-clear",
    "--tls.guess",
    "--geometry.remove",
    "false",
    "--no-warnings",
    "-X",
    "never",
)


def compile_osm(osm_file: str | Path, out_dir: str | Path, name: str = "bengaluru") -> Path:
    """Run ``netconvert`` on a clipped OSM file."""
    from .. import sumo_env

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{name}.net.xml"
    if target.exists() and target.stat().st_size > 0:
        return target
    cmd = [
        sumo_env.netconvert_binary(),
        "--osm-files",
        str(osm_file),
        "-o",
        str(target),
        *OSM_NETCONVERT_FLAGS,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not target.exists():
        raise RuntimeError(f"netconvert failed:\n{result.stdout}\n{result.stderr}")
    return target


def build(
    area_name: str = "central",
    out_dir: str | Path = "data/networks",
    data_dir: str | Path | None = None,
    allow_download: bool = True,
    progress: bool = True,
) -> tuple[RoadNetwork, Path, BBox]:
    """Full pipeline: extract -> compile -> load. Returns graph, ``.net.xml`` and bbox."""
    bbox = area(area_name)
    name = f"bengaluru_{area_name}"
    pbf = source_pbf(data_dir, allow_download=allow_download)
    if progress:
        h, w = bbox.size_km
        print(f"Bengaluru / {area_name}: {h} km x {w} km, bbox={bbox.as_netconvert()}")
    osm_file = extract_osm(pbf, bbox, Path(out_dir) / "osm", name=name, progress=progress)
    net_file = compile_osm(osm_file, Path(out_dir) / "net", name=name)
    network = load_net_file(net_file)
    return network, net_file, bbox


def nearest_node(network: RoadNetwork, lat: float, lon: float) -> int:
    """Index of the network node closest to a geographic point.

    An equirectangular correction is applied so that the metric is
    approximately true over a few kilometres.
    """
    geo = network.lonlat()
    if not network.has_georeference():
        coords = network.coordinates()
        scale = 0.0
        d = (coords[:, 0] - lon) ** 2 + (coords[:, 1] - lat) ** 2
        return int(np.argmin(d))
    lon0 = float(geo[:, 0].mean())
    lat0 = float(geo[:, 1].mean())
    scale = 111.320 * float(np.cos(np.radians(lat0)))
    d = (geo[:, 1] - lat) ** 2 + ((geo[:, 0] - lon) * scale) ** 2
    return int(np.argmin(d))


def landmark_nodes(network: RoadNetwork, landmarks: list[str] | None = None) -> dict[str, int]:
    """Map landmark names to the nearest node in the loaded network."""
    names = landmarks or LANDMARKS_BY_AREA.get("central", list(LANDMARKS))
    return {name: nearest_node(network, *LANDMARKS[name]) for name in names if name in LANDMARKS}
