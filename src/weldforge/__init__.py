"""WeldForge -- from 3D model to robotic weld.

A planning toolchain for robotic arc welding:

* :mod:`weldforge.cad` reads a STEP, IGES or mesh file and measures it;
* :mod:`weldforge.seams` recognises the weld seams and classifies the joints;
* :mod:`weldforge.wps` generates a welding procedure for each seam and
  translates it into setpoints for a specific power source;
* :mod:`weldforge.fixture` decides where and which way up the part goes;
* :mod:`weldforge.kin` solves the robot's kinematics;
* :mod:`weldforge.path` builds the torch path and orders the seams;
* :mod:`weldforge.collide` checks the torch and the arm against the cell;
* :mod:`weldforge.sim` reports reachability, collisions, cycle time and cost;
* :mod:`weldforge.post` writes the program for the controller in the cell;
* :mod:`weldforge.plan` wires those together;
* :mod:`weldforge.api` exposes the lot over REST for the CAD plugins.

Units are millimetres, radians and seconds throughout -- see
:mod:`weldforge.units`, which is also the only sanctioned place to convert.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
