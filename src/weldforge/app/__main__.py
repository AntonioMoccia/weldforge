"""``python -m weldforge.app`` -- start the studio.

The ``weldforge-studio`` entry point is the normal way in, but it only exists
once the package is installed, and it is a *GUI* script: on Windows it runs
under ``pythonw.exe`` with no console, so nothing it prints can be seen. This
module starts the same window from an interpreter that still has a terminal
attached -- which is what the one-command launcher uses, and what to reach for
when something fails at start-up.
"""

from __future__ import annotations

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main())
