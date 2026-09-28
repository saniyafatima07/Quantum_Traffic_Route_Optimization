"""The demo's HTTP server.

Deliberately stdlib-only.  A web framework would be the second-largest
dependency in the project for a page that is served from ``localhost``, and
adding one to *show* the results would mean the results depend on a package
that has nothing to do with producing them.

Two shapes of client are supported, because the demo is useful in both:

*   a browser, polling ``/api/state`` and then pulling ``/api/data``; and
*   a script, doing the same with ``curl`` — so a reviewer who distrusts the
    page can still get every number it displays.

The build runs on a background thread, so the page shows progress instead of a
blank screen for the couple of minutes a nine-method sweep over SUMO takes.
"""

from __future__ import annotations

import gzip
import json
import threading
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .pipeline import Comparison, run_comparison

STATIC = Path(__file__).parent / "static"
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json",
}


@dataclass
class BuildState:
    """Progress of the comparison build, readable while it is still running."""

    status: str = "idle"
    stage: str = ""
    fraction: float = 0.0
    error: str | None = None
    ready: bool = False
    methods_done: list[str] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stage": self.stage,
            "fraction": round(self.fraction, 4),
            "error": self.error,
            "ready": self.ready,
            "methods_done": self.methods_done,
        }


class DemoService:
    """Owns the comparison: builds it once, then serves it to every client."""

    def __init__(self, builder, cache: str | Path = "data/demo/comparison.json") -> None:
        self._builder = builder
        self._cache = Path(cache)
        self.state = BuildState()
        self.comparison: Comparison | None = None
        self._lock = threading.Lock()

    def cached(self) -> Comparison | None:
        """The saved comparison, if there is a usable one.

        A cache that will not open is reported rather than ignored.  Swallowing
        the error looks like it works — the page simply takes three minutes to
        appear — and the user is left guessing whether the comparison is running
        or hung.
        """
        if not self._cache.is_file():
            return None
        try:
            return Comparison.load(self._cache)
        except Exception as exc:  # noqa: BLE001 - any breakage means rebuild
            self.state.status = "rebuilding"
            self.state.error = f"cache at {self._cache} could not be read ({exc}); rebuilding"
            return None

    def adopt(self, comparison: Comparison) -> None:
        with self._lock:
            self.comparison = comparison
            self.state.methods_done = [r.name for r in comparison.runs]
            self.state.status = "ready"
            self.state.ready = True
            self.state.fraction = 1.0
            self.state.stage = f"{len(comparison.runs)} methods ready"

    def start(self) -> None:
        if self.state.status == "building":
            return
        cached = self.cached()
        if cached is not None:
            self.adopt(cached)
            return
        notice = self.state.error
        self.state = BuildState(status="building", stage="starting", fraction=0.0, error=notice)
        threading.Thread(target=self._build, name="qitransit-demo", daemon=True).start()

    def rebuild(self) -> None:
        """Throw the cached comparison away and start over."""
        self._cache.unlink(missing_ok=True)
        with self._lock:
            self.comparison = None
        self._builder.invalidate()
        self.state = BuildState(status="building", stage="starting", fraction=0.0)
        threading.Thread(target=self._build, name="qitransit-demo", daemon=True).start()

    def _build(self) -> None:
        def progress(stage: str, fraction: float) -> None:
            self.state.stage = stage
            self.state.fraction = fraction

        try:
            comparison = run_comparison(**self._builder.kwargs(), progress=progress)
        except Exception as exc:  # noqa: BLE001 - the page must show why, not hang
            self.state.status = "failed"
            self.state.error = f"{type(exc).__name__}: {exc}"
            self.state.stage = traceback.format_exc(limit=4)
            return
        comparison.save(self._cache)
        self.adopt(comparison)

    def payload(self) -> dict | None:
        with self._lock:
            return self.comparison.as_dict() if self.comparison else None


class Builder:
    """The keyword arguments for :func:`run_comparison`, deferred.

    Building the network takes ten seconds and most of a gigabyte, and loading
    SUMO's bindings is not free either.  Doing that work at argument-parse time
    would make ``--help`` feel broken and turn any import error into a traceback
    before the user had asked for anything.  Deferring it to the build thread
    keeps start-up instant and leaves the failure somewhere it can be shown.
    """

    def __init__(self, prepare) -> None:
        self._prepare = prepare
        self._resolved: dict | None = None

    def kwargs(self) -> dict:
        if self._resolved is None:
            self._resolved = self._prepare()
        return dict(self._resolved)

    def invalidate(self) -> None:
        self._resolved = None


class Handler(BaseHTTPRequestHandler):
    service: DemoService
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = urlparse(self.path).path
        if path == "/api/state":
            self._json(self.service.state.snapshot())
        elif path == "/api/data":
            payload = self.service.payload()
            if payload is None:
                self._json({"error": "comparison not ready"}, status=503)
            else:
                self._json(payload, compress=True)
        elif path.startswith("/static/"):
            self._static(path.removeprefix("/static/"))
        elif path == "/":
            self._static("index.html")
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = urlparse(self.path).path
        if path == "/api/rebuild":
            self.service.rebuild()
            self._json({"status": "building"})
        elif path == "/api/sumo-gui":
            self._sumo_gui()
        else:
            self.send_error(404)

    def _sumo_gui(self) -> None:
        """Open the selected solution in SUMO's own graphical client.

        The headless numbers in the table come from TraCI; this is the
        other half of the claim, the part where a person watches the routes
        being driven and can disagree with them.
        """
        import subprocess
        import sys

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        net_file = body.get("net_file", "")
        routes_file = body.get("routes_file", "")
        if not (net_file and routes_file and Path(routes_file).is_file()):
            self._json({"error": "no routes file for that method"}, status=400)
            return
        from ..sumo_env import _executable

        try:
            gui = _executable("sumo-gui")
        except FileNotFoundError as exc:
            self._json({"error": str(exc)}, status=503)
            return
        subprocess.Popen(
            [gui, "-n", net_file, "-r", routes_file, "--start"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._json({"launched": True})

    def _static(self, name: str) -> None:
        target = (STATIC / name).resolve()
        if not str(target).startswith(str(STATIC.resolve())) or not target.is_file():
            self.send_error(404)
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200, compress: bool = False) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        encoding = "identity"
        # Only compress for a client that said it can decompress.  A payload
        # labelled gzip to something that did not ask for it is not a large
        # response, it is a parse error — and the client most likely to hit that
        # is the reviewer curling /api/data to check the page's arithmetic.
        if compress and len(body) > 4096 and "gzip" in self.headers.get("Accept-Encoding", ""):
            body = gzip.compress(body, 6)
            encoding = "gzip"
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Encoding", encoding)
        self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        return None


def serve(service: DemoService, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Start the server and the background build, and return the server."""
    handler = type("BoundHandler", (Handler,), {"service": service})
    httpd = ThreadingHTTPServer((host, port), handler)
    service.start()
    return httpd
