"""WeldForge Studio: the desktop application.

Layered so that almost all of it is testable without a display:

``project``     the document and its ``.weldforge`` file
``scene3d``     the VTK scene and picking -- runs offscreen
``tools``       the interactive tools, as state machines with no Qt
``panels``      the Qt dock panels -- run headless
``send``        getting the program to the machine
``main_window`` the shell, which is the only part that needs a screen
"""

from .project import (
    PROJECT_SUFFIX,
    Project,
    ProjectError,
    RecentFiles,
    SentRecord,
    attach_bench,
    attach_obstacle,
    attach_part,
    cell_for,
    clear_bench,
    load_project,
    materialise_bench,
    remove_obstacle,
    save_project,
)
from .send import SendError, SendTarget, dispatch

__all__ = [
    "PROJECT_SUFFIX",
    "Project",
    "ProjectError",
    "RecentFiles",
    "SentRecord",
    "attach_part",
    "attach_bench",
    "attach_obstacle",
    "remove_obstacle",
    "cell_for",
    "materialise_bench",
    "clear_bench",
    "load_project",
    "save_project",
    "SendError",
    "SendTarget",
    "dispatch",
    "main",
]


def main(argv: list[str] | None = None) -> int:
    """Start WeldForge Studio. Imported lazily so the core needs no Qt."""
    from .main_window import main as _main

    return _main(argv)
