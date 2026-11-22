"""Joint-type classification.

The groove angle alone is not enough. Three joints that all measure 90 deg on
the weld side are welded differently and need different procedures:

* a **fillet** in the inside corner of a single bent or machined body,
* a **tee**, where the end of one plate lands on the broad face of another,
* a **lap**, where two plates lie face to face and the weld runs along the
  upper plate's edge.

So classification takes a :class:`JointContext` -- groove angle and concavity
from :mod:`weldforge.seams.dihedral`, plus how the two bodies actually touch,
how thick they are, and how big the faces are -- and returns the joint type
with a confidence the operator can sort by.

Band edges follow ISO 9692-1 practice rather than round numbers: a 60-135 deg
groove is the fillet band because that is the range a standard torch angle and
a symmetric leg still make sense in; outside it the joint needs preparation or
a different weld type entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..model import JointType

logger = logging.getLogger(__name__)

__all__ = [
    "ContactKind",
    "JointContext",
    "Classification",
    "classify",
    "GROOVE_BANDS",
]


class ContactKind:
    """How two bodies meet, as determined by the contact finder."""

    NONE = "none"  # same body, or no contact found
    SHARED_TOPOLOGY = "shared"  # fused weldment: the faces share an edge
    FACE_CONTACT = "face_contact"  # antiparallel faces touching (a landing)
    PARALLEL_OVERLAP = "parallel_overlap"  # plates stacked face to face: a lap
    COPLANAR_GAP = "coplanar_gap"  # side-by-side faces in one plane: a butt
    NEAR = "near"  # within tolerance but not characterised


# Weld-side groove bands, in degrees. Open intervals on the upper bound.
GROOVE_BANDS: dict[str, tuple[float, float]] = {
    "slit": (0.0, 30.0),  # faces nearly parallel -> edge weld
    "acute": (30.0, 60.0),  # prepared V, or a sharp corner
    "fillet": (60.0, 135.0),  # the fillet band
    "flat": (135.0, 215.0),  # effectively coplanar -> butt
    "open": (215.0, 330.0),  # outside corner
    "reflex": (330.0, 360.0),  # degenerate: a fold back on itself
}


def _band(groove_deg: float) -> str:
    for name, (lo, hi) in GROOVE_BANDS.items():
        if lo <= groove_deg < hi:
            return name
    return "reflex"


def _band_centrality(groove_deg: float, band: str) -> float:
    """1.0 at the middle of the band, falling to 0.0 at its edges."""
    lo, hi = GROOVE_BANDS[band]
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    if half <= 0.0:
        return 0.0
    return float(max(0.0, 1.0 - abs(groove_deg - mid) / half))


@dataclass
class JointContext:
    """Everything the classifier is allowed to look at."""

    groove_rad: float
    concavity: int = 0
    groove_spread_rad: float = 0.0
    same_solid: bool = False
    contact_kind: str = ContactKind.NONE
    gap_mm: float = 0.0
    face_a_type: str = "planar"
    face_b_type: str = "planar"
    thickness_a_mm: float = 0.0
    thickness_b_mm: float = 0.0
    area_a_mm2: float = 0.0
    area_b_mm2: float = 0.0
    seam_length_mm: float = 0.0
    closed: bool = False
    exact_geometry: bool = True  # False for a mesh import
    is_blend_face: bool = False  # one flank is a small fillet/chamfer blend
    # True when the landing member reaches the edge of the receiving face, so
    # the receiving plate does not continue past the footprint. That is an L
    # (corner) configuration; a T has material on both sides of the footprint.
    landing_flush_boundary: bool = False

    @property
    def groove_deg(self) -> float:
        return float(np.rad2deg(self.groove_rad))

    @property
    def min_thickness_mm(self) -> float:
        pos = [t for t in (self.thickness_a_mm, self.thickness_b_mm) if t > 0.0]
        return min(pos) if pos else 0.0


@dataclass
class Classification:
    joint_type: JointType
    confidence: float
    band: str = ""
    notes: list[str] = field(default_factory=list)
    rejected_reason: str = ""

    @property
    def weldable(self) -> bool:
        return self.joint_type.is_weldable


def classify(ctx: JointContext) -> Classification:
    """Assign a joint type and a confidence in ``[0, 1]``."""
    groove = ctx.groove_deg
    band = _band(groove)
    notes: list[str] = []

    # -- hard rejections -----------------------------------------------------

    if ctx.is_blend_face:
        # A chamfer or a fillet radius between two faces produces a pair of
        # plausible-looking grooves that are machined features, not joints.
        return Classification(JointType.UNKNOWN, 0.0, band, notes, "flank is a blend/chamfer face")

    if band == "reflex":
        return Classification(
            JointType.UNKNOWN, 0.0, band, notes, f"groove {groove:.0f} deg folds back"
        )

    if ctx.same_solid and ctx.concavity < 0 and band in ("open", "flat"):
        # A convex edge of one body is the outside of the part: nothing to weld.
        return Classification(JointType.UNKNOWN, 0.0, band, notes, "convex edge of a single body")

    if ctx.same_solid and ctx.concavity == 0:
        # A tangent edge inside one body, e.g. where a plate's face is split by
        # a construction edge. Not a joint.
        return Classification(
            JointType.UNKNOWN, 0.0, band, notes, "tangent edge inside a single body"
        )

    # -- type by band and context -------------------------------------------

    joint = JointType.UNKNOWN

    if band == "slit":
        joint = JointType.EDGE
        notes.append("faces nearly parallel: edge/flange weld")

    elif band == "acute":
        if ctx.contact_kind == ContactKind.COPLANAR_GAP:
            joint = JointType.BUTT
            notes.append("prepared groove: bevel angle below 60 deg")
        else:
            joint = JointType.FILLET
            notes.append(f"acute groove {groove:.0f} deg: torch access will be tight")

    elif band == "fillet":
        if ctx.closed and ctx.contact_kind in (
            ContactKind.FACE_CONTACT,
            ContactKind.PARALLEL_OVERLAP,
        ):
            joint = JointType.PLUG
            notes.append("closed seam on a covered opening: plug/slot weld")
        elif ctx.same_solid:
            joint = JointType.FILLET
            notes.append("inside corner of a single body")
        elif ctx.contact_kind == ContactKind.PARALLEL_OVERLAP:
            joint = JointType.LAP
            notes.append("plates face to face: lap weld along the upper edge")
        elif ctx.contact_kind in (ContactKind.FACE_CONTACT, ContactKind.SHARED_TOPOLOGY):
            if ctx.landing_flush_boundary:
                joint = JointType.FILLET
                notes.append(
                    "member sits on the edge of the receiving face: L/corner "
                    "configuration, welded as a fillet"
                )
            else:
                joint = JointType.TEE
                notes.append("member lands on a face: T joint, welded as a fillet")
        else:
            joint = JointType.FILLET

    elif band == "flat":
        if ctx.same_solid:
            joint = JointType.UNKNOWN
            return Classification(joint, 0.0, band, notes, "coplanar faces of a single body")
        joint = JointType.BUTT
        notes.append("coplanar members: butt weld")
        if ctx.gap_mm > 0.0:
            notes.append(f"root gap {ctx.gap_mm:.1f} mm")

    elif band == "open":
        joint = JointType.CORNER
        notes.append("outside corner joint")

    # -- confidence ----------------------------------------------------------

    confidence = 0.35 + 0.35 * _band_centrality(groove, band)

    # A groove that wanders along the seam means one parameter set will not fit.
    spread_deg = float(np.rad2deg(ctx.groove_spread_rad))
    if spread_deg > 20.0:
        confidence -= 0.20
        notes.append(f"groove varies by {spread_deg:.0f} deg along the seam")
    elif spread_deg < 2.0:
        confidence += 0.10

    # Contact that was actually characterised is stronger evidence than mere
    # proximity.
    if ctx.contact_kind in (
        ContactKind.SHARED_TOPOLOGY,
        ContactKind.FACE_CONTACT,
        ContactKind.PARALLEL_OVERLAP,
        ContactKind.COPLANAR_GAP,
    ):
        confidence += 0.10
    elif ctx.contact_kind == ContactKind.NEAR:
        confidence -= 0.10
        notes.append("bodies are close but the contact type is unclear")

    if not ctx.exact_geometry:
        # A tessellation cannot tell 90.0 deg from 89.3 deg.
        confidence -= 0.20
        notes.append("mesh geometry: angles are approximate")

    if ctx.min_thickness_mm <= 0.0:
        confidence -= 0.05
        notes.append("member thickness unknown")

    # Very short seams are usually artefacts of a split face.
    if 0.0 < ctx.seam_length_mm < 5.0:
        confidence -= 0.15
        notes.append(f"seam only {ctx.seam_length_mm:.1f} mm long")

    if ctx.gap_mm > 3.0 and joint is JointType.BUTT:
        confidence -= 0.15
        notes.append(f"root gap {ctx.gap_mm:.1f} mm exceeds 3 mm: backing may be needed")

    confidence = float(np.clip(confidence, 0.0, 1.0))
    return Classification(joint, round(confidence, 3), band, notes)
