"""Shared location of the bundled ``data/`` tree.

Re-exported from :mod:`weldforge.wps.tables` so the kinematics package does not
import the welding-procedure package just to find a directory.
"""

from __future__ import annotations

from ..wps.tables import DATA_DIR

__all__ = ["DATA_DIR"]
