"""ABB RAPID output.

RAPID is the richest target: ABB's arc-welding package has first-class weld
instructions (``ArcLStart`` / ``ArcL`` / ``ArcLEnd``) that take ``seamdata``
and ``welddata`` records, so the procedure's numbers go into the program as
data rather than as digital-output fiddling. Ignition, burn-back and crater
fill are handled by the controller.

Units: millimetres and quaternions, which is what ``robtarget`` carries, so no
scaling is needed. Orientation goes out as ``(q1, q2, q3, q4)`` in ABB's
``w, x, y, z`` order -- **not** the ``x, y, z, w`` order used inside WeldForge.
"""

from __future__ import annotations

import numpy as np

from ..geom import transform as tf
from ..model import PathPoint, WeldJob, WeldPass
from .base import Postprocessor, PostResult, header_lines
from .registry import register

__all__ = ["RapidPost"]


@register
class RapidPost(Postprocessor):
    key = "rapid"
    target = "ABB RAPID (IRC5 / OmniCore, arc welding package)"
    units = "mm"
    extension = "mod"
    comment = "!"

    def _robtarget(self, point: PathPoint, name: str) -> str:
        position = tf.translation_of(point.pose)
        # ABB quaternion order is (w, x, y, z); ours is (x, y, z, w).
        x, y, z, w = tf.matrix_to_quat(point.pose)
        precision = self.options.position_precision
        angle_precision = self.options.angle_precision
        joints = (
            ", ".join(f"{np.rad2deg(v):.{angle_precision}f}" for v in point.joints[:6])
            if point.joints is not None and len(point.joints) >= 6
            else "0, 0, 0, 0, 0, 0"
        )
        return (
            f"  CONST robtarget {name} := "
            f"[[{position[0]:.{precision}f}, {position[1]:.{precision}f}, "
            f"{position[2]:.{precision}f}], "
            f"[{w:.{angle_precision}f}, {x:.{angle_precision}f}, "
            f"{y:.{angle_precision}f}, {z:.{angle_precision}f}], "
            f"[0, 0, 0, 0], {self._extjoint(point)}];"
            f"  {self.comment} joints: {joints}"
        )

    def _extjoint(self, point: PathPoint) -> str:
        """``extjoint`` for a robtarget: the positioner axes, 9E9 where unused.

        The fourth element of a robtarget carries the external axes, named
        eax_a..eax_f. Left at 9E9 ("not used") the controller welds at
        whatever orientation the faceplate happens to be in -- which is the
        orientation the *previous* seam left it in.

        Rotary external axes go out in degrees, matching the controller's own
        convention.
        """
        if not point.external_axes:
            return "[9E9, 9E9, 9E9, 9E9, 9E9, 9E9]"
        values = [
            f"{np.rad2deg(point.external_axes[name]):.4f}" for name in sorted(point.external_axes)
        ]
        values += ["9E9"] * (6 - len(values))
        return "[" + ", ".join(values[:6]) + "]"

    def _welddata(self, weld_pass: WeldPass, name: str) -> list[str]:
        """``seamdata`` and ``welddata`` for one pass.

        ``weld_speed`` in RAPID is mm/s, which is what the procedure already
        carries. Weaving goes into ``weavedata``; a pass without weaving gets
        the zero record so the instruction signature stays uniform.
        """
        lines = [
            f"  CONST seamdata sm_{name} := "
            f"[0, {weld_pass.arc_start_delay_s:.2f}, "
            f"[0, 0, 0, 0, 0, 0, 0, 0, 0], 0, 0, 0, 0, 0, "
            f"[0, 0, 0, 0, 0, 0, 0, 0, 0], 0, 0, 0, "
            f"[0, 0, 0, 0, 0, 0, 0, 0, 0], {weld_pass.crater_fill_s:.2f}, "
            f"[0, 0, 0, 0, 0, 0, 0, 0, 0], {weld_pass.burnback_s:.2f}, 0, 0];",
            f"  CONST welddata wd_{name} := "
            f"[{weld_pass.travel_speed_mm_s:.2f}, 0, "
            f"[{weld_pass.voltage_v:.1f}, 0, {weld_pass.wire_speed_m_min:.2f}, "
            f"{weld_pass.current_a:.0f}, 0, 0, 0, 0, 0], "
            f"[0, 0, 0, 0, 0, 0, 0, 0, 0]];",
        ]
        if weld_pass.weave.active:
            lines.append(
                f"  CONST weavedata wv_{name} := "
                f"[1, 0, {weld_pass.weave.amplitude_mm:.2f}, "
                f"{weld_pass.weave.frequency_hz:.2f}, 0, 0, 0, "
                f"{weld_pass.weave.dwell_ms / 1000.0:.3f}, "
                f"{weld_pass.weave.dwell_ms / 1000.0:.3f}, 0, 0, 0, 0];"
            )
        else:
            lines.append(f"  CONST weavedata wv_{name} := [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];")
        return lines

    def generate(self, job: WeldJob) -> PostResult:
        result = PostResult(target=self.target)
        opts = self.options
        name = opts.program_name

        targets: list[str] = []
        welddata: list[str] = []
        moves: list[str] = []

        for index, path in enumerate(job.paths):
            points, skipped = self.usable_points(path)
            result.n_skipped += skipped
            if len(points) < 2:
                result.warnings.append(
                    f"seam {path.seam_id} pass {path.pass_index + 1}: "
                    "too few reachable stations, skipped"
                )
                continue

            weld_pass = self.pass_for(job, path)
            tag = f"s{index:03d}"
            if weld_pass is not None:
                welddata.extend(self._welddata(weld_pass, tag))

            if opts.include_comments:
                moves.append(f"    {self.comment} {self.seam_comment(job, path)}")

            weld_points = [p for p in points if p.weld_on]
            for station, point in enumerate(points):
                target_name = f"p_{tag}_{station:04d}"
                targets.append(self._robtarget(point, target_name))

                zone = "fine" if point.weld_on else "z10"
                tool = f"{opts.tool_name}, \\WObj:={opts.workobject_name}"

                if not point.weld_on:
                    speed = f"v{int(max(10, point.speed_mm_s * 10))}"
                    instruction = "MoveJ" if point.motion.value == "joint" else "MoveL"
                    moves.append(f"    {instruction} {target_name}, {speed}, {zone}, {tool};")
                elif weld_pass is None:
                    moves.append(f"    MoveL {target_name}, v100, {zone}, {tool};")
                elif point is weld_points[0]:
                    moves.append(
                        f"    ArcLStart {target_name}, v100, sm_{tag}, wd_{tag}, "
                        f"\\Weave:=wv_{tag}, {zone}, {tool};"
                    )
                elif point is weld_points[-1]:
                    moves.append(
                        f"    ArcLEnd {target_name}, v100, sm_{tag}, wd_{tag}, "
                        f"\\Weave:=wv_{tag}, {zone}, {tool};"
                    )
                else:
                    moves.append(
                        f"    ArcL {target_name}, v100, sm_{tag}, wd_{tag}, "
                        f"\\Weave:=wv_{tag}, {zone}, {tool};"
                    )
                result.n_moves += 1

        lines: list[str] = []
        if opts.include_header:
            lines.extend(header_lines(job, opts, self.comment))
        lines.append("")
        lines.append(f"MODULE {name}")
        lines.append("")
        lines.extend(welddata)
        lines.append("")
        lines.extend(targets)
        lines.append("")
        if any(p.external_axes for path in job.paths for p in path.points):
            names = sorted(
                {name for path in job.paths for p in path.points for name in p.external_axes}
            )
            lines.append(f"  {self.comment} external axes, eax_a onwards: {', '.join(names)}")
            lines.append(
                f"  {self.comment} Confirm this mapping against the controller's own "
                "mechanical unit definition."
            )
            lines.append("")
        lines.append("  PROC main()")
        lines.append(f"    {self.comment} Move to a known posture before anything else.")
        lines.append("    MoveAbsJ jpos_home, v200, fine, " + opts.tool_name + ";")
        lines.extend(moves)
        lines.append("    MoveAbsJ jpos_home, v200, fine, " + opts.tool_name + ";")
        lines.append("  ENDPROC")
        lines.append("")
        lines.append("ENDMODULE")

        result.files[f"{name}.{self.extension}"] = "\n".join(lines) + "\n"
        return result
