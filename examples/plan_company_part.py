"""Plan the real customer weldment end to end, and write every program format.

Run it as:

.. code-block:: bash

    python examples/plan_company_part.py "../Giunto  ver 2.0.stp"

What it demonstrates, in order: reading a 58-body STEP assembly, recognising
its seams by contact (no two bodies share topology, so nothing else would find
them), generating a procedure per seam, placing the part on the bench, building
the torch paths with configuration-consistent inverse kinematics, simulating
the cycle, and writing programs for five controllers.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from weldforge.fixture import cell_robot, load_cell
from weldforge.plan import PlanOptions, plan_job
from weldforge.post import PostOptions, available_targets, get_postprocessor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", help="STEP file")
    parser.add_argument("--cell", default="bench_cell")
    parser.add_argument("--material", default="S355")
    parser.add_argument("--out", default="out")
    parser.add_argument(
        "--max-seams",
        type=int,
        default=25,
        help="cap the seams carried into motion planning (longest first)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="  %(message)s",
    )

    path = Path(args.part)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 1

    cell = load_cell(args.cell)
    robot = cell_robot(cell)
    print(f"cell  : {cell.name} / {robot.name} (reach {robot.reach_mm:.0f} mm)")
    print(f"part  : {path.name}")

    options = PlanOptions(name=path.stem)
    options.procedure.base_material = args.material
    options.max_seams = args.max_seams
    options.min_seam_length_mm = 15.0
    # A real 200-seam weldment does not need a fine placement search for a
    # first pass; raise these for a final plan.
    options.placement.grid_steps = 3
    options.placement.yaw_steps = 4
    options.placement.max_orientations = 4

    started = time.perf_counter()
    job = plan_job(path, cell, options, robot)
    elapsed = time.perf_counter() - started

    print(f"\nplanned in {elapsed:.1f}s")
    print(f"  timings        : {job.metadata.get('timings')}")

    detection = job.metadata.get("detection", {})
    print(f"\nseam recognition")
    print(f"  contact fragments     {detection.get('contact_fragments')}")
    print(f"  topological fragments {detection.get('topological_fragments')}")
    print(f"  duplicates removed    {detection.get('duplicates_removed')}")
    print(f"  seams after chaining  {detection.get('seams_after_chaining')}")
    if "seams_truncated" in job.metadata:
        truncated = job.metadata["seams_truncated"]
        print(f"  NOTE: {truncated['found']} found, {truncated['kept']} kept for motion")

    print(f"\nseams carried forward: {len(job.seams)}")
    print(f"  by joint type  : {dict(Counter(s.joint_type.value for s in job.seams))}")
    print(f"  by position    : {dict(Counter(s.dominant_position.value for s in job.seams))}")
    print(f"  weld length    : {job.total_weld_length_mm:.0f} mm")
    thicknesses = sorted(
        {round(t, 1) for s in job.seams for t in (s.thickness_a_mm, s.thickness_b_mm) if t > 0}
    )
    print(f"  thicknesses    : {thicknesses} mm")

    if job.placement is not None:
        placement = job.placement.summary()
        print(f"\nplacement")
        print(f"  position       : {placement['translation_mm']} mm")
        print(f"  orientation    : rpy {placement['rpy_deg']} deg")
        print(f"  score          : {placement['score']:.3f}  ({placement['detail']})")
        print(f"  welding        : {placement['position_mix']}")
        print(f"  reachable      : {placement['n_reachable']}/{len(job.seams)}")
        for violation in placement["violations"]:
            print(f"  ! {violation}")

    report = job.report
    print(f"\nverdict: {'OK' if report.ok else 'PROBLEMS'}")
    print(f"  reachable      : {report.reachable}")
    print(f"  collision free : {report.collision_free}")
    print(
        f"  cycle time     : {report.cycle_time_s:.0f} s "
        f"(arc {report.arc_time_s:.0f} s, duty {report.duty_cycle:.0%})"
    )
    print(
        f"  consumables    : {report.wire_consumption_kg * 1000:.0f} g wire, "
        f"{report.gas_consumption_l:.0f} l gas"
    )
    print(f"  passes planned : {len(job.paths)}")
    for error in report.errors[:8]:
        print(f"  ERROR   {error}")
    for warning in report.warnings[:8]:
        print(f"  warning {warning}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nprograms written to {out_dir}/")
    for target in available_targets():
        post = get_postprocessor(target, PostOptions(program_name="GIUNTO"))
        result = post.generate(job)
        for filename, text in result.files.items():
            (out_dir / filename).write_text(text)
            size = len(text.encode())
            print(f"  {target:9s} {filename:32s} {size / 1024:6.1f} kB  {result.n_moves} moves")

    return 0 if report.ok else 2


if __name__ == "__main__":
    sys.exit(main())
