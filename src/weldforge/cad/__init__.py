"""CAD import: OpenCASCADE backend adapter, topology extraction, loaders."""

from .backend import CadBackend, CadBackendError, backend_available, backend_name, get_backend
from .loader import BREP_SUFFIXES, MESH_SUFFIXES, LoadOptions, load_mesh_part, load_part
from .topology import (
    FaceEvaluator,
    RawTopology,
    RayCaster,
    estimate_thickness,
    extract,
    face_min_extent,
    sample_edge,
)

__all__ = [
    "CadBackend",
    "CadBackendError",
    "backend_available",
    "backend_name",
    "get_backend",
    "LoadOptions",
    "load_part",
    "load_mesh_part",
    "BREP_SUFFIXES",
    "MESH_SUFFIXES",
    "FaceEvaluator",
    "RawTopology",
    "RayCaster",
    "extract",
    "sample_edge",
    "estimate_thickness",
    "face_min_extent",
]
