"""The seam recognition entry point.

Runs both detectors -- shared topology and body contact -- then reconciles
their output:

1. **De-duplicate.** A fused weldment with an imprinted contact can be found
   twice. Duplicates are matched geometrically, not by id, and the more
   confident one survives.
2. **Chain.** Fragments that are really one weld are merged
   (:mod:`weldforge.seams.chain`).
3. **Order and label.** Longest first, with a human label an operator can
   refer to on the shop floor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..model import JointType, Part, Seam
from .chain import ChainOptions, chain_seams
from .contact_detect import ContactOptions, detect_contact_seams, find_contacts
from .topo_detect import TopoDetectOptions, detect_topological_seams

logger = logging.getLogger(__name__)

__all__ = ["DetectOptions", "DetectionReport", "detect_seams", "deduplicate_seams"]


@dataclass
class DetectOptions:
    topo: TopoDetectOptions = field(default_factory=TopoDetectOptions)
    contact: ContactOptions = field(default_factory=ContactOptions)
    chain: ChainOptions = field(default_factory=ChainOptions)
    use_topological: bool = True
    use_contact: bool = True
    dedupe_tol_mm: float = 0.5
    # Fraction of stations that must coincide for two seams to be the same weld.
    dedupe_overlap: float = 0.8
    min_length_mm: float = 5.0
    min_confidence: float = 0.25
    label: bool = True

    @classmethod
    def for_assembly(cls) -> DetectOptions:
        """Tuned for a multi-body assembly export: contact does the work."""
        return cls(use_topological=True, use_contact=True)

    @classmethod
    def for_single_body(cls) -> DetectOptions:
        """Tuned for one machined or bent body: no contacts to find."""
        return cls(use_topological=True, use_contact=False)


@dataclass
class DetectionReport:
    """What the detectors found, and what they threw away."""

    n_topological: int = 0
    n_contact: int = 0
    n_duplicates: int = 0
    n_after_chaining: int = 0
    n_dropped_short: int = 0
    n_dropped_confidence: int = 0
    contacts: list = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> dict:
        return {
            "topological_fragments": self.n_topological,
            "contact_fragments": self.n_contact,
            "duplicates_removed": self.n_duplicates,
            "seams_after_chaining": self.n_after_chaining,
            "dropped_too_short": self.n_dropped_short,
            "dropped_low_confidence": self.n_dropped_confidence,
            "contacts": [c.summary() for c in self.contacts],
            "seconds": round(self.seconds, 3),
        }


def explain_no_seams(part, report: DetectionReport, options=None) -> str:
    """Why a detection run found nothing, in one sentence.

    "0 seams" on its own reads as a broken importer, and the most common cause
    is not a bug at all: a STEP exported as a single part rather than as the
    assembly. Contact detection compares bodies, so with one body there is no
    joint to find, however good the geometry is. The reason is read off the
    report rather than guessed, so it can only ever say what actually
    happened.
    """
    if report.n_after_chaining > 0 or report.n_topological or report.n_contact:
        if report.n_dropped_short or report.n_dropped_confidence:
            limits = ""
            if options is not None:
                limits = (
                    f" (shorter than {options.min_length_mm:g} mm, or below "
                    f"{options.min_confidence:g} confidence)"
                )
            return (
                f"candidates were found but all were filtered out{limits}: "
                f"{report.n_dropped_short} too short, "
                f"{report.n_dropped_confidence} too uncertain"
            )
        return "candidates were found but none survived chaining"

    if len(part.solids) <= 1:
        return (
            f"{part.name} is a single body, so there is no joint between parts to "
            "detect. Export the assembly from your CAD seat as STEP -- one body per "
            "part -- or trace the weld line by hand"
        )
    if part.metadata.get("geometry_kind") == "mesh":
        return (
            "no contact or edge pair qualified. This is a mesh, where a facet normal "
            "cannot tell 90.0 degrees from 89.3; import STEP for exact geometry"
        )
    return (
        f"no contact or edge pair qualified across {len(part.solids)} bodies. "
        "Check that the bodies actually touch in the model, or trace the weld line "
        "by hand"
    )


# ---------------------------------------------------------------------------
# de-duplication
# ---------------------------------------------------------------------------


def _overlap_fraction(a: Seam, b: Seam, tol: float) -> float:
    """Fraction of *a*'s stations lying within *tol* of any of *b*'s."""
    if a.n_points == 0 or b.n_points == 0:
        return 0.0
    distances = np.linalg.norm(a.points[:, None, :] - b.points[None, :, :], axis=2)
    return float(np.mean(distances.min(axis=1) <= tol))


def deduplicate_seams(
    seams: list[Seam], tol_mm: float = 0.5, min_overlap: float = 0.8
) -> tuple[list[Seam], int]:
    """Drop seams that duplicate a better one; returns ``(kept, n_removed)``.

    Two seams count as the same weld when most of one's stations sit on the
    other. The survivor is chosen by confidence, then by length -- a detector
    that saw the whole seam beats one that saw half of it.
    """
    order = sorted(
        range(len(seams)),
        key=lambda i: (-seams[i].confidence, -seams[i].length_mm),
    )
    kept: list[Seam] = []
    removed = 0
    for i in order:
        candidate = seams[i]
        duplicate = False
        for existing in kept:
            # Cheap rejection first: centroids far apart cannot overlap.
            if (
                np.linalg.norm(candidate.centroid - existing.centroid)
                > (candidate.length_mm + existing.length_mm) / 2.0 + tol_mm
            ):
                continue
            forward = _overlap_fraction(candidate, existing, tol_mm)
            backward = _overlap_fraction(existing, candidate, tol_mm)
            if max(forward, backward) >= min_overlap:
                duplicate = True
                break
        if duplicate:
            removed += 1
        else:
            kept.append(candidate)
    return kept, removed


# ---------------------------------------------------------------------------
# labelling
# ---------------------------------------------------------------------------

_JOINT_LABEL_IT = {
    JointType.FILLET: "cordone d'angolo",
    JointType.TEE: "giunto a T",
    JointType.LAP: "sovrapposizione",
    JointType.BUTT: "testa a testa",
    JointType.CORNER: "spigolo esterno",
    JointType.EDGE: "lembo",
    JointType.PLUG: "tappo",
    JointType.UNKNOWN: "non classificato",
}


def _label(seam: Seam, index: int) -> str:
    """Shop-floor label: number, joint type, length and position."""
    kind = _JOINT_LABEL_IT.get(seam.joint_type, seam.joint_type.value)
    return f"{index:02d} - {kind} {seam.length_mm:.0f} mm ({seam.dominant_position.value})"


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def detect_seams(
    part: Part,
    options: DetectOptions | None = None,
) -> tuple[list[Seam], DetectionReport]:
    """Recognise every weld seam in *part*.

    Returns the seams, longest first, and a report of what was discarded --
    the report is what lets an operator ask "why did it not find my weld?"
    instead of guessing.
    """
    import time

    opts = options or DetectOptions()
    report = DetectionReport()
    t0 = time.perf_counter()

    fragments: list[Seam] = []

    if opts.use_contact:
        contacts = find_contacts(part, opts.contact)
        report.contacts = contacts
        contact_seams = detect_contact_seams(part, opts.contact, contacts)
        report.n_contact = len(contact_seams)
        fragments.extend(contact_seams)

    if opts.use_topological:
        topo_seams = detect_topological_seams(part, opts.topo)
        report.n_topological = len(topo_seams)
        fragments.extend(topo_seams)

    fragments, report.n_duplicates = deduplicate_seams(
        fragments, opts.dedupe_tol_mm, opts.dedupe_overlap
    )

    chain_opts = opts.chain
    chain_opts.min_length_mm = max(chain_opts.min_length_mm, opts.min_length_mm)
    seams = chain_seams(fragments, chain_opts)
    report.n_after_chaining = len(seams)

    before = len(seams)
    seams = [s for s in seams if s.length_mm >= opts.min_length_mm]
    report.n_dropped_short = before - len(seams)

    before = len(seams)
    seams = [s for s in seams if s.confidence >= opts.min_confidence]
    report.n_dropped_confidence = before - len(seams)

    seams.sort(key=lambda s: -s.length_mm)
    if opts.label:
        for i, seam in enumerate(seams, start=1):
            seam.label = _label(seam, i)

    report.seconds = time.perf_counter() - t0
    logger.info(
        "detect_seams: %d seams, %.0f mm total, %.2fs",
        len(seams),
        sum(s.length_mm for s in seams),
        report.seconds,
    )
    return seams, report
