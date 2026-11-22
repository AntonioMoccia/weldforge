"""Weld seam recognition: dihedral geometry, two detectors, chaining, classification."""

from .chain import ChainOptions, chain_seams
from .classify import Classification, ContactKind, JointContext, classify
from .contact_detect import (
    ContactOptions,
    ContactPair,
    detect_contact_seams,
    find_contacts,
)
from .detect import DetectionReport, DetectOptions, deduplicate_seams, detect_seams
from .dihedral import (
    FaceSide,
    InwardMode,
    SeamGeometry,
    compute_seam_geometry,
    groove_angle,
    joint_bisector,
    material_angle,
)
from .topo_detect import TopoDetectOptions, detect_topological_seams

__all__ = [
    "ChainOptions",
    "chain_seams",
    "Classification",
    "ContactKind",
    "JointContext",
    "classify",
    "ContactOptions",
    "ContactPair",
    "detect_contact_seams",
    "find_contacts",
    "DetectOptions",
    "DetectionReport",
    "detect_seams",
    "deduplicate_seams",
    "FaceSide",
    "InwardMode",
    "SeamGeometry",
    "compute_seam_geometry",
    "groove_angle",
    "joint_bisector",
    "material_angle",
    "TopoDetectOptions",
    "detect_topological_seams",
]
