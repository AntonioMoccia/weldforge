"""Universal Robots URScript output.

The only target that works in **metres**, and the one where the unit boundary
earns its keep: every length is divided by 1000 exactly once, here.

Orientation goes out as a rotation vector (axis times angle), which is what
``movel`` and ``movej`` take in a pose. Welding is driven through digital and
analog I/O, since a UR has no welding instruction set; the tool-side I/O is
used where a torch is mounted on the tool flange.
"""

from __future__ import annotations

from ..geom import transform as tf
from ..model import PathPoint, WeldJob, WeldPass
from .base import Postprocessor, PostResult, header_lines
from .registry import register

__all__ = ["UrScriptPost"]


@register
class UrScriptPost(Postprocessor):
    key = "urscript"
    target = "Universal Robots URScript (e-Series / PolyScope)"
    units = "m"
    extension = "script"
    comment = "#"

    def _pose(self, point: PathPoint) -> str:
        """``p[x, y, z, rx, ry, rz]`` in metres and a rotation vector."""
        position = tf.translation_of(point.pose) / 1000.0
        axis, angle = tf.axis_angle_of(point.pose)
        rotation = axis * angle
        return (
            f"p[{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}, "
            f"{rotation[0]:.6f}, {rotation[1]:.6f}, {rotation[2]:.6f}]"
        )

    def generate(self, job: WeldJob) -> PostResult:
        result = PostResult(target=self.target)
        opts = self.options
        lines: list[str] = []

        if opts.include_header:
            lines.extend(header_lines(job, opts, self.comment))

        lines += [
            "",
            f"def {opts.program_name.lower()}():",
            f"  {self.comment} Tool centre point: the wire tip, in metres.",
            "  set_tcp(p[0, 0, 0, 0, 0, 0])",
            f"  {self.comment} Replace with the measured torch TCP before running.",
            "  set_payload(2.5)",
            "",
            "  def weld_on(current, voltage, wire_speed):",
            f"    {self.comment} Analog references are 0-10 V on a UR; scale to the",
            f"    {self.comment} power source's full scale (see data/machines/*.yaml).",
            f"    set_analog_out(0, current / {_CURRENT_FULL_SCALE:.1f})",
            f"    set_analog_out(1, voltage / {_VOLTAGE_FULL_SCALE:.1f})",
            f"    set_digital_out({_GAS_OUT}, True)",
            "    sleep(0.3)",
            f"    set_digital_out({_ARC_OUT}, True)",
            f"    {self.comment} Wait for the power source to confirm current flow.",
            f"    while get_digital_in({_ARC_OK_IN}) == False:",
            "      sleep(0.01)",
            "    end",
            "  end",
            "",
            "  def weld_off(crater_s):",
            "    sleep(crater_s)",
            f"    set_digital_out({_ARC_OUT}, False)",
            "    sleep(0.1)",
            f"    set_digital_out({_GAS_OUT}, False)",
            "    set_analog_out(0, 0.0)",
            "    set_analog_out(1, 0.0)",
            "  end",
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

            weld_pass: WeldPass | None = self.pass_for(job, path)
            if opts.include_comments:
                lines.append(f"  {self.comment} {self.seam_comment(job, path)}")

            weld_points = [p for p in points if p.weld_on]
            for point in points:
                if point.weld_on and weld_pass is not None and point is weld_points[0]:
                    lines.append(
                        f"  weld_on({weld_pass.current_a:.0f}, "
                        f"{weld_pass.voltage_v:.1f}, {weld_pass.wire_speed_m_min:.2f})"
                    )

                speed = point.speed_mm_s / 1000.0
                if point.motion.value == "joint" and not point.weld_on:
                    lines.append(f"  movej({self._pose(point)}, a=1.2, v={speed:.4f})")
                else:
                    blend = 0.0 if point.weld_on else 0.005
                    lines.append(
                        f"  movel({self._pose(point)}, a=0.5, v={speed:.4f}, r={blend:.4f})"
                    )
                result.n_moves += 1

                if point.weld_on and weld_pass is not None and point is weld_points[-1]:
                    lines.append(f"  weld_off({weld_pass.crater_fill_s:.2f})")

            lines.append("")

        lines += [
            "end",
            "",
            f"{opts.program_name.lower()}()",
        ]
        result.files[f"{opts.program_name}.{self.extension}"] = "\n".join(lines) + "\n"
        return result


_ARC_OUT = 0
_GAS_OUT = 1
_ARC_OK_IN = 0
_CURRENT_FULL_SCALE = 500.0
_VOLTAGE_FULL_SCALE = 50.0
