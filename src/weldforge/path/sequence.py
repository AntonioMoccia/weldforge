"""Seam ordering.

Two things fight each other and both matter:

**Cycle time.** Air moves between seams are pure waste. On a weldment with
dozens of short seams they can exceed the arc time, so the order should keep
the torch near where it just finished -- a travelling-salesman problem over
seam endpoints, with the extra freedom that a seam may be welded in either
direction.

**Distortion.** Welding is local heating, and heating one side of a structure
and then the other pulls it out of shape. Shop practice is to balance the heat:
alternate sides of the part, work outwards from the middle, and avoid piling
consecutive welds into one region. The cheapest route in travel terms is
usually the worst one for distortion, because it works along one side and then
comes back down the other.

So the cost minimised here is both:

.. code-block:: text

    cost = air_travel_mm + distortion_weight * imbalance_penalty

with the weight exposed so a job can be tuned either way. A thin fabrication
that will banana needs distortion to dominate; a stiff machined block does not
care and should be welded fast.

Ordering is greedy nearest-neighbour followed by 2-opt, which on the few
hundred seams a weldment has converges in well under a second and gets within
a few percent of optimal -- precision that is not worth more time, since the
real travel also depends on the robot's own joint motion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..model import Seam

logger = logging.getLogger(__name__)

__all__ = ["SequenceOptions", "SequencedSeam", "sequence_seams", "air_travel_mm"]


@dataclass
class SequenceOptions:
    """Weighting and switches for the sequencer."""

    # Millimetres of air travel considered equivalent to one unit of heat
    # imbalance. 0 optimises purely for cycle time.
    distortion_weight: float = 150.0
    allow_reverse: bool = True
    two_opt_passes: int = 4
    # Weld the longest seams first within a balanced ordering: they do the most
    # structural work and locking the assembly early limits later movement.
    prefer_long_first: bool = False
    start_point: np.ndarray | None = None

    @classmethod
    def fast(cls) -> SequenceOptions:
        """Cycle time only -- for a stiff part that will not move."""
        return cls(distortion_weight=0.0)

    @classmethod
    def low_distortion(cls) -> SequenceOptions:
        """Heat balance dominates -- for thin or slender fabrications."""
        return cls(distortion_weight=600.0)


@dataclass
class SequencedSeam:
    """A seam in welding order, possibly reversed."""

    seam: Seam
    reversed: bool = False
    air_travel_mm: float = 0.0

    @property
    def start(self) -> np.ndarray:
        return self.seam.end if self.reversed else self.seam.start

    @property
    def end(self) -> np.ndarray:
        return self.seam.start if self.reversed else self.seam.end

    def oriented(self) -> Seam:
        """The seam as it should be welded."""
        return self.seam.reversed() if self.reversed else self.seam


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------


def _imbalance_penalty(order: list[SequencedSeam], centroid: np.ndarray) -> float:
    """How lopsided the heat history is, as a running sum.

    After each seam, the signed sum of where heat has gone (relative to the
    part centroid) is accumulated. A sequence that alternates sides keeps the
    running sum near zero; one that does all of one side first builds a large
    sum, and that is exactly the sequence that bends the part.

    Normalised by the number of seams so the weight means the same thing on a
    5-seam part and a 200-seam one.
    """
    if not order:
        return 0.0
    running = np.zeros(3)
    total = 0.0
    for item in order:
        offset = item.seam.centroid - centroid
        norm = float(np.linalg.norm(offset))
        if norm > 1e-9:
            running = running + offset / norm * item.seam.length_mm
        total += float(np.linalg.norm(running))
    return total / len(order)


def air_travel_mm(order: list[SequencedSeam], start: np.ndarray | None = None) -> float:
    """Total straight-line air travel for a sequence."""
    if not order:
        return 0.0
    total = 0.0
    position = start if start is not None else order[0].start
    for item in order:
        total += float(np.linalg.norm(item.start - position))
        position = item.end
    return total


def _total_cost(
    order: list[SequencedSeam],
    centroid: np.ndarray,
    options: SequenceOptions,
) -> float:
    cost = air_travel_mm(order, options.start_point)
    if options.distortion_weight > 0.0:
        cost += options.distortion_weight * _imbalance_penalty(order, centroid)
    return cost


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------


def sequence_seams(
    seams: list[Seam],
    options: SequenceOptions | None = None,
) -> list[SequencedSeam]:
    """Order *seams* for welding, choosing a direction for each."""
    opts = options or SequenceOptions()
    if not seams:
        return []
    if len(seams) == 1:
        return [SequencedSeam(seams[0])]

    centroid = np.mean([s.centroid for s in seams], axis=0)

    # --- greedy nearest neighbour ------------------------------------------
    remaining = list(seams)
    if opts.prefer_long_first:
        remaining.sort(key=lambda s: -s.length_mm)

    order: list[SequencedSeam] = []
    position = opts.start_point if opts.start_point is not None else remaining[0].start
    running = np.zeros(3)

    while remaining:
        best_index = 0
        best_reverse = False
        best_cost = np.inf
        for index, seam in enumerate(remaining):
            directions = (False, True) if opts.allow_reverse else (False,)
            for reverse in directions:
                start = seam.end if reverse else seam.start
                travel = float(np.linalg.norm(start - position))
                cost = travel
                if opts.distortion_weight > 0.0:
                    offset = seam.centroid - centroid
                    norm = float(np.linalg.norm(offset))
                    direction = offset / norm if norm > 1e-9 else np.zeros(3)
                    projected = running + direction * seam.length_mm
                    cost += opts.distortion_weight * float(np.linalg.norm(projected)) / len(seams)
                if cost < best_cost:
                    best_cost, best_index, best_reverse = cost, index, reverse

        seam = remaining.pop(best_index)
        item = SequencedSeam(seam, best_reverse)
        item.air_travel_mm = float(np.linalg.norm(item.start - position))
        order.append(item)
        position = item.end
        offset = seam.centroid - centroid
        norm = float(np.linalg.norm(offset))
        if norm > 1e-9:
            running = running + offset / norm * seam.length_mm

    # --- 2-opt ------------------------------------------------------------
    best_cost = _total_cost(order, centroid, opts)
    for _ in range(max(0, opts.two_opt_passes)):
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                candidate = order[:i] + order[i : j + 1][::-1] + order[j + 1 :]
                cost = _total_cost(candidate, centroid, opts)
                if cost < best_cost - 1e-9:
                    order, best_cost, improved = candidate, cost, True
        if not improved:
            break

    # Direction can be re-chosen once the order is settled.
    if opts.allow_reverse:
        position = opts.start_point if opts.start_point is not None else order[0].start
        for item in order:
            forward = float(np.linalg.norm(item.seam.start - position))
            backward = float(np.linalg.norm(item.seam.end - position))
            item.reversed = backward < forward
            item.air_travel_mm = min(forward, backward)
            position = item.end

    logger.info(
        "sequence_seams: %d seams, %.0f mm air travel, cost %.0f",
        len(order),
        air_travel_mm(order, opts.start_point),
        best_cost,
    )
    return order
