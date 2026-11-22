"""KUKA KRL output.

KRL has no standard arc-welding instruction set: welding is done with a
technology package (ArcTech) or, more commonly on a retrofitted cell, by
driving the power source over digital and analog outputs. This generator takes
the second route, because it works on any KR C4/C5 without a licence -- and
emits the setpoints as named variables so swapping in ArcTech later means
editing one subprogram, not every move.

Units: millimetres and **Z-Y-X Euler angles in degrees** (KUKA's A, B, C --
A about Z, B about Y, C about X). WeldForge stores roll-pitch-yaw in the same
intrinsic Z-Y-X order, so A=yaw, B=pitch, C=roll, converted to degrees.
"""

from __future__ import annotations

import numpy as np

from ..geom import transform as tf
from ..model import PathPoint, WeldJob, WeldPass
from .base import Postprocessor, PostResult, header_lines
from .registry import register

__all__ = ["KrlPost"]


@register
class KrlPost(Postprocessor):
    key = "krl"
    target = "KUKA KRL (KR C4 / C5, digital weld interface)"
    units = "mm"
    extension = "src"
    comment = ";"

    def _frame(self, point: PathPoint) -> str:
        position = tf.translation_of(point.pose)
        roll, pitch, yaw = tf.matrix_to_rpy(point.pose)
        precision = self.options.position_precision
        # External axes are E1..E6 inside the same frame literal. Omitting
        # them leaves the positioner wherever the previous seam put it, so the
        # part gets welded in the wrong orientation.
        external = ""
        if point.external_axes:
            parts = [
                f"E{index + 1} {np.rad2deg(point.external_axes[name]):.3f}"
                for index, name in enumerate(sorted(point.external_axes))
            ]
            external = ", " + ", ".join(parts)
        return (
            f"{{X {position[0]:.{precision}f}, Y {position[1]:.{precision}f}, "
            f"Z {position[2]:.{precision}f}, "
            f"A {np.rad2deg(yaw):.3f}, B {np.rad2deg(pitch):.3f}, "
            f"C {np.rad2deg(roll):.3f}{external}}}"
        )

    def _weld_on(self, weld_pass: WeldPass) -> list[str]:
        """Set the power source up and strike the arc."""
        return [
            f"  {self.comment} --- weld parameters: {weld_pass.current_a:.0f} A, "
            f"{weld_pass.voltage_v:.1f} V, {weld_pass.wire_speed_m_min:.2f} m/min",
            f"  WELD_CURRENT = {weld_pass.current_a:.1f}",
            f"  WELD_VOLTAGE = {weld_pass.voltage_v:.1f}",
            f"  WELD_WIRESPEED = {weld_pass.wire_speed_m_min:.2f}",
            f"  WELD_JOB = {weld_pass.index + 1}",
            "  SET_WELD_PARAMS()",
            f"  $OUT[{_GAS_OUT}] = TRUE   {self.comment} gas pre-flow",
            f"  WAIT SEC {weld_pass.arc_start_delay_s:.2f}",
            f"  $OUT[{_ARC_OUT}] = TRUE   {self.comment} arc on",
            f"  WAIT FOR $IN[{_ARC_OK_IN}]   {self.comment} current flowing",
        ]

    def _weld_off(self, weld_pass: WeldPass) -> list[str]:
        return [
            f"  WAIT SEC {weld_pass.crater_fill_s:.2f}   {self.comment} crater fill",
            f"  $OUT[{_ARC_OUT}] = FALSE  {self.comment} arc off",
            f"  WAIT SEC {weld_pass.burnback_s:.2f}   {self.comment} burn-back",
            f"  $OUT[{_GAS_OUT}] = FALSE  {self.comment} gas post-flow off",
        ]

    def generate(self, job: WeldJob) -> PostResult:
        result = PostResult(target=self.target)
        opts = self.options
        lines: list[str] = []

        if opts.include_header:
            lines.extend(header_lines(job, opts, self.comment))

        lines += [
            "",
            f"DEF {opts.program_name}()",
            "",
            "DECL REAL WELD_CURRENT, WELD_VOLTAGE, WELD_WIRESPEED",
            "DECL INT WELD_JOB",
            "",
            f"  {self.comment} Standard KUKA initialisation goes here (BAS, interrupts).",
            *(
                [
                    f"  {self.comment} Positioner axes are emitted as E1.. in each frame;",
                    f"  {self.comment} they need the external kinematic ($EX_AX_NUM)",
                    f"  {self.comment} configured and the axis order confirmed.",
                ]
                if any(p.external_axes for path in job.paths for p in path.points)
                else []
            ),
            "  BAS(#INITMOV, 0)",
            f"  $TOOL = TOOL_DATA[{_TOOL_INDEX}]",
            f"  $BASE = BASE_DATA[{_BASE_INDEX}]",
            "  $VEL.CP = 0.5",
            "  PTP $AXIS_ACT",
            "  PTP HOME",
            "",
        ]

        for path in job.paths:
            points, skipped = self.usable_points(path)
            result.n_skipped += skipped
            if len(points) < 2:
                result.warnings.append(
                    f"seam {path.seam_id} pass {path.pass_index + 1}: "
                    "too few reachable stations, skipped"
                )
                continue

            weld_pass = self.pass_for(job, path)
            if opts.include_comments:
                lines.append(f"  {self.comment} {self.seam_comment(job, path)}")

            weld_points = [p for p in points if p.weld_on]
            for point in points:
                if point.weld_on and weld_pass is not None and point is weld_points[0]:
                    lines.extend(self._weld_on(weld_pass))

                if point.weld_on:
                    speed = max(0.001, point.speed_mm_s / 1000.0)  # KRL wants m/s
                    lines.append(f"  $VEL.CP = {speed:.4f}")
                    lines.append(f"  LIN {self._frame(point)} C_VEL")
                else:
                    lines.append("  $VEL.CP = 0.25")
                    instruction = "PTP" if point.motion.value == "joint" else "LIN"
                    lines.append(f"  {instruction} {self._frame(point)} C_DIS")
                result.n_moves += 1

                if point.weld_on and weld_pass is not None and point is weld_points[-1]:
                    lines.extend(self._weld_off(weld_pass))

            lines.append("")

        lines += [
            "  PTP HOME",
            "",
            "END",
            "",
            f"{self.comment} --- helper ---------------------------------------------------",
            f"{self.comment} SET_WELD_PARAMS writes the analog words for the power source.",
            f"{self.comment} Scaling is cell-specific: see data/machines/*.yaml for the",
            f"{self.comment} mapping this job was planned with.",
            "DEF SET_WELD_PARAMS()",
            f"  $ANOUT[{_CURRENT_AOUT}] = WELD_CURRENT / {_CURRENT_FULL_SCALE:.1f}",
            f"  $ANOUT[{_VOLTAGE_AOUT}] = WELD_VOLTAGE / {_VOLTAGE_FULL_SCALE:.1f}",
            "END",
        ]

        result.files[f"{opts.program_name}.{self.extension}"] = "\n".join(lines) + "\n"
        return result


# Signal assignments. These are cell wiring, not robot programming: they are
# named here so a commissioning engineer can find and change them in one place.
_ARC_OUT = 1
_GAS_OUT = 2
_ARC_OK_IN = 1
_TOOL_INDEX = 1
_BASE_INDEX = 1
_CURRENT_AOUT = 1
_VOLTAGE_AOUT = 2
_CURRENT_FULL_SCALE = 500.0
_VOLTAGE_FULL_SCALE = 50.0
