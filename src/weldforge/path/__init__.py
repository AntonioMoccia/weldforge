"""Torch path generation: poses, weaving, sequencing and path assembly."""

from .build import PathBuildReport, PathOptions, build_job_paths, build_pass_path
from .sequence import SequencedSeam, SequenceOptions, air_travel_mm, sequence_seams
from .torch import (
    across_direction,
    approach_point,
    measure_travel_angle,
    measure_work_angle,
    retract_point,
    torch_pose,
    torch_poses_for_pass,
    weave_offset,
    wire_direction,
)

__all__ = [
    "PathBuildReport",
    "PathOptions",
    "build_job_paths",
    "build_pass_path",
    "SequencedSeam",
    "SequenceOptions",
    "air_travel_mm",
    "sequence_seams",
    "across_direction",
    "approach_point",
    "measure_travel_angle",
    "measure_work_angle",
    "retract_point",
    "torch_pose",
    "torch_poses_for_pass",
    "weave_offset",
    "wire_direction",
]
