"""Locate the SUMO installation and make its Python bindings importable.

Eclipse SUMO ships ``traci`` and ``sumolib`` as part of the distribution
instead of publishing them to PyPI, so the project virtual environment cannot
see them.  Importing this module appends ``$SUMO_HOME/tools`` to ``sys.path``,
which is all that is required to drive a simulation.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from pathlib import Path

DEFAULT_SUMO_HOME = Path("/usr/share/sumo")
_SEARCH_DIRS = (
    DEFAULT_SUMO_HOME,
    Path("/usr/local/share/sumo"),
    Path("/opt/sumo"),
    Path("/Applications/SUMO.app/Contents/Resources"),
)


def sumo_home() -> Path:
    """Return the SUMO installation root, honouring ``SUMO_HOME``."""
    override = os.environ.get("SUMO_HOME")
    if override:
        candidate = Path(override).expanduser()
        if (candidate / "tools").is_dir():
            return candidate
    for candidate in _SEARCH_DIRS:
        if (candidate / "tools").is_dir():
            return candidate
    return DEFAULT_SUMO_HOME


def _executable(name: str) -> str:
    """Resolve a SUMO command-line tool, preferring ``SUMO_HOME/bin``."""
    local = sumo_home() / "bin" / name
    if local.is_file():
        return str(local)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"SUMO executable {name!r} not found. Install Eclipse SUMO or add its "
        f"bin/ directory to PATH."
    )


@functools.cache
def sumo_binary() -> str:
    """The microscopic simulator, ``sumo``."""
    return _executable("sumo")


@functools.cache
def netconvert_binary() -> str:
    """The network importer/builder, ``netconvert``."""
    return _executable("netconvert")


@functools.cache
def duarouter_binary() -> str:
    """The standalone router, ``duarouter``."""
    return _executable("duarouter")


def sumo_version() -> str:
    """Version string reported by the ``sumo`` binary."""
    out = subprocess.run(
        [sumo_binary(), "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in out.splitlines():
        lowered = line.lower()
        if "sumo" in lowered and (" 1." in lowered or " 2." in lowered):
            return line.strip()
    return out.splitlines()[0].strip() if out else "unknown"


def ensure_bindings() -> Path:
    """Put ``$SUMO_HOME/tools`` on ``sys.path`` and return it."""
    import sys

    tools = sumo_home() / "tools"
    if tools.is_dir():
        entry = str(tools)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return tools


def import_traci():
    """Import and return the ``traci`` module, wiring up ``sys.path`` first."""
    ensure_bindings()
    import traci  # noqa: PLC0415  (deliberately deferred)

    return traci


def import_sumolib():
    """Import and return the ``sumolib`` module, wiring up ``sys.path`` first."""
    ensure_bindings()
    import sumolib  # noqa: PLC0415  (deliberately deferred)

    return sumolib


def available() -> bool:
    """True when both the binaries and the Python bindings can be located."""
    try:
        sumo_binary()
        netconvert_binary()
        ensure_bindings()
        import_traci()
    except Exception:
        return False
    return True
