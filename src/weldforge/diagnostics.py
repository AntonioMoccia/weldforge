"""Logging to a file, and a report of what this installation can do.

Why this module exists
----------------------
``weldforge-studio`` is a *GUI* entry point, so on Windows it launches under
``pythonw.exe``, which has no console: anything written to stdout or stderr is
discarded by the operating system. A traceback that would have explained a
failed CAD import is therefore invisible to the one person who needs it. So
the application logs to a file as well, always, from the first line of
``main``.

The same report is reachable without a display through ``weldforge doctor``,
which is what to ask an operator to run when an import fails on their machine
and not on ours.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import platform
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "log_dir",
    "log_file",
    "setup_logging",
    "report",
    "format_report",
    "DEPENDENCIES",
]

# (import name, distribution name, what breaks without it)
DEPENDENCIES: tuple[tuple[str, str, str], ...] = (
    ("numpy", "numpy", "everything"),
    ("scipy", "scipy", "kinematics, fitting"),
    ("trimesh", "trimesh", "STL/OBJ/PLY import"),
    ("networkx", "networkx", "seam grouping"),
    ("yaml", "pyyaml", "cell, robot and procedure data"),
    ("pydantic", "pydantic", "project validation"),
    ("OCP", "cadquery-ocp", "STEP/IGES/BREP import"),
    ("PySide6", "PySide6", "the desktop window"),
    ("vtk", "vtk", "the 3-D view"),
)


def log_dir() -> Path:
    """Per-user directory for the log files, created on demand.

    Windows puts it under ``%LOCALAPPDATA%`` because that is where a user can
    find it without administrator rights and where a roaming profile will not
    copy it around the network.
    """
    override = os.environ.get("WELDFORGE_LOG_DIR")
    if override:
        base = Path(override)
    elif sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        base = Path(root) / "WeldForge" / "logs"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Logs" / "WeldForge"
    else:
        root = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
        base = Path(root) / "weldforge" / "logs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def log_file(name: str = "studio") -> Path:
    return log_dir() / f"{name}.log"


def setup_logging(
    name: str = "studio",
    level: int = logging.INFO,
    console: bool = True,
) -> Path:
    """Send the root logger to a rotating file, and to the console if there is one.

    Returns the path of the log file so the application can show it in an
    error dialog -- an error the operator can send us is worth more than one
    they can only describe.
    """
    path = log_file(name)
    root = logging.getLogger()
    root.setLevel(min(level, logging.DEBUG))

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # The file gets DEBUG: it costs nothing and it is the only record we get
    # of a failure on a machine we cannot reach.
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    handler.set_name("weldforge_file")
    _replace_handler(root, handler)

    if console and (sys.stderr is not None and sys.stderr.isatty()):
        stream = logging.StreamHandler()
        stream.setLevel(level)
        stream.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        stream.set_name("weldforge_console")
        _replace_handler(root, stream)

    logger.info("--- WeldForge starting ---")
    for line in format_report(report()).splitlines():
        logger.info("%s", line)
    return path


def _replace_handler(root: logging.Logger, handler: logging.Handler) -> None:
    """Install *handler*, dropping an earlier one of the same name.

    ``setup_logging`` is idempotent so that calling it twice -- the CLI
    starting the studio, say -- does not double every log line.
    """
    for existing in list(root.handlers):
        if existing.get_name() == handler.get_name():
            root.removeHandler(existing)
            existing.close()
    root.addHandler(handler)


def report() -> dict[str, Any]:
    """Everything worth knowing about this installation, as plain data."""
    from . import __version__

    packages: dict[str, dict[str, Any]] = {}
    for module_name, dist, purpose in DEPENDENCIES:
        entry: dict[str, Any] = {"purpose": purpose, "distribution": dist}
        try:
            import_module(module_name)
            entry["importable"] = True
        except Exception as exc:
            entry["importable"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        try:
            entry["version"] = version(dist)
        except PackageNotFoundError:
            entry["version"] = None
        packages[module_name] = entry

    backend: str | None
    try:
        from .cad import backend_name

        backend = backend_name()
    except Exception as exc:  # pragma: no cover - import-time breakage
        backend = None
        logger.debug("backend_name() failed: %s", exc)

    from .cad.loader import BREP_SUFFIXES, MESH_SUFFIXES

    return {
        "weldforge": __version__,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "frozen": getattr(sys, "frozen", False),
        "cad_backend": backend,
        "brep_formats": sorted(BREP_SUFFIXES) if backend else [],
        "mesh_formats": sorted(MESH_SUFFIXES),
        "log_file": str(log_file()),
        "packages": packages,
    }


def format_report(data: dict[str, Any]) -> str:
    """The report as aligned text, for a terminal or a log file."""
    lines = [
        f"WeldForge        : {data['weldforge']}",
        f"Python           : {data['python']}  ({data['python_executable']})",
        f"Platform         : {data['platform']}",
        f"Frozen build     : {'yes' if data['frozen'] else 'no'}",
        f"CAD backend      : {data['cad_backend'] or 'NONE -- STEP/IGES cannot be read'}",
        f"B-rep formats    : {', '.join(data['brep_formats']) or '(none)'}",
        f"Mesh formats     : {', '.join(data['mesh_formats'])}",
        f"Log file         : {data['log_file']}",
        "",
        "Dependencies",
    ]
    width = max(len(name) for name in data["packages"])
    for name, entry in data["packages"].items():
        mark = "ok  " if entry["importable"] else "MISS"
        ver = entry["version"] or "-"
        line = f"  {mark} {name.ljust(width)}  {ver.ljust(10)} {entry['purpose']}"
        if not entry["importable"]:
            line += f"\n       -> {entry.get('error', 'not installed')}"
            line += f"\n       -> pip install {entry['distribution']}"
        lines.append(line)
    return "\n".join(lines)
