"""FANUC TP output, as a readable ``.LS`` listing.

A ``.LS`` file is the ASCII form of a TP program; it is loaded onto the
controller and compiled to ``.TP`` there (or by ROBOGUIDE). Positions live in
the ``/POS`` section as ``P[n]`` records and the instruction list references
them -- which is why the generator builds both and keeps their numbering in
step.

Units: millimetres with W-P-R orientation in degrees. FANUC's W, P, R are
rotations about X, Y and Z respectively, applied in that order -- the same
intrinsic Z-Y-X composition WeldForge stores, so ``W = roll``, ``P = pitch``,
``R = yaw``.

Arc control uses the Arc Tool package's ``Arc Start`` / ``Arc End`` with weld
schedules. The schedule numbers are placeholders keyed to the pass index; they
must be matched to the schedules actually stored in the controller.
"""

from __future__ import annotations

import numpy as np

from ..geom import transform as tf
from ..model import PathPoint, WeldJob, WeldPass
from .base import Postprocessor, PostResult, header_lines
from .registry import register

__all__ = ["FanucPost"]


@register
class FanucPost(Postprocessor):
    key = "fanuc"
    target = "FANUC TP listing (.LS, Arc Tool)"
    units = "mm"
    extension = "ls"
    comment = "!"

    def _position_record(self, index: int, point: PathPoint, group: int = 1) -> list[str]:
        position = tf.translation_of(point.pose)
        roll, pitch, yaw = tf.matrix_to_rpy(point.pose)
        lines = [
            f"P[{index}]{{",
            f"   GP{group}:",
            "\tUF : 1, UT : 1,\t\tCONFIG : 'N U T, 0, 0, 0',",
            f"\tX = {position[0]:9.3f}  mm,\tY = {position[1]:9.3f}  mm,"
            f"\tZ = {position[2]:9.3f}  mm,",
            f"\tW = {np.rad2deg(roll):9.3f} deg,\tP = {np.rad2deg(pitch):9.3f} deg,"
            f"\tR = {np.rad2deg(yaw):9.3f} deg",
        ]
        # A positioner is a second motion group on a FANUC controller, so its
        # axes get their own GP block. Leaving them out welds the part at
        # whatever orientation the previous seam left the faceplate in.
        if point.external_axes:
            lines.append(f"   GP{group + 1}:")
            lines.append("\tUF : 0, UT : 0,")
            for i, name in enumerate(sorted(point.external_axes)):
                suffix = "," if i < len(point.external_axes) - 1 else ""
                lines.append(
                    f"\tJ{i + 1} = {np.rad2deg(point.external_axes[name]):9.3f} deg{suffix}"
                )
        lines.append("};")
        return lines

    def generate(self, job: WeldJob) -> PostResult:
        result = PostResult(target=self.target)
        opts = self.options

        instructions: list[str] = []
        positions: list[str] = []
        line_number = 0
        position_index = 0

        def emit(text: str) -> None:
            nonlocal line_number
            line_number += 1
            instructions.append(f"{line_number:4d}:  {text}")

        emit("UFRAME_NUM=1 ;")
        emit("UTOOL_NUM=1 ;")
        emit("!Move to home before starting ;")
        emit("J P[1] 50% FINE ;")

        # P[1] is home: the first station of the first path, backed off.
        if job.paths and job.paths[0].points:
            position_index += 1
            positions.extend(self._position_record(position_index, job.paths[0].points[0]))

        for path in job.paths:
            points, skipped = self.usable_points(path)
            result.n_skipped += skipped
            if len(points) < 2:
                result.warnings.append(
                    f"seam {path.seam_id} pass {path.pass_index + 1}: "
                    "too few reachable stations, skipped"
                )
                continue

            weld_pass: WeldPass | None = self.pass_for(job, path)
            schedule = (weld_pass.index + 1) if weld_pass is not None else 1
            if opts.include_comments:
                emit(f"!{self.seam_comment(job, path)[:60]} ;")

            weld_points = [p for p in points if p.weld_on]
            for point in points:
                position_index += 1
                positions.extend(self._position_record(position_index, point))
                target = f"P[{position_index}]"

                if not point.weld_on:
                    motion = "J" if point.motion.value == "joint" else "L"
                    speed = "50%" if motion == "J" else f"{int(max(10, point.speed_mm_s))}mm/sec"
                    emit(f"{motion} {target} {speed} CNT50 ;")
                else:
                    speed = f"{max(1, int(round(point.speed_mm_s * 60)))}mm/min"
                    if point is weld_points[0]:
                        emit(f"L {target} {speed} FINE Arc Start[{schedule}] ;")
                    elif point is weld_points[-1]:
                        emit(f"L {target} {speed} FINE Arc End[{schedule}] ;")
                    else:
                        emit(f"L {target} {speed} CNT10 ;")
                result.n_moves += 1

            if weld_pass is not None and weld_pass.weave.active:
                emit(
                    f"!weave {weld_pass.weave.shape.value} "
                    f"{weld_pass.weave.amplitude_mm:.1f}mm "
                    f"{weld_pass.weave.frequency_hz:.1f}Hz ;"
                )

        emit("J P[1] 50% FINE ;")

        lines: list[str] = []
        if opts.include_header:
            lines.extend(header_lines(job, opts, self.comment))
        lines += [
            "/PROG  " + opts.program_name[:8].upper(),
            "/ATTR",
            "OWNER\t\t= MNEDITOR;",
            'COMMENT\t\t= "WeldForge";',
            "PROG_SIZE\t= 0;",
            "CREATE\t\t= DATE 00-00-00  TIME 00:00:00;",
            "MODIFIED\t= DATE 00-00-00  TIME 00:00:00;",
            "FILE_NAME\t= ;",
            "VERSION\t\t= 0;",
            f"LINE_COUNT\t= {line_number};",
            "MEMORY_SIZE\t= 0;",
            "PROTECT\t\t= READ_WRITE;",
            "TCD:  STACK_SIZE\t= 0,",
            "      TASK_PRIORITY\t= 50,",
            "      TIME_SLICE\t= 0,",
            "      BUSY_LAMP_OFF\t= 0,",
            "      ABORT_REQUEST\t= 0,",
            "      PAUSE_REQUEST\t= 0;",
            "DEFAULT_GROUP\t= 1,*,*,*,*;",
            "CONTROL_CODE\t= 00000000 00000000;",
            "/MN",
        ]
        lines.extend(instructions)
        lines.append("/POS")
        lines.extend(positions)
        lines.append("/END")

        result.files[f"{opts.program_name[:8].upper()}.{self.extension}"] = "\n".join(lines) + "\n"
        return result
