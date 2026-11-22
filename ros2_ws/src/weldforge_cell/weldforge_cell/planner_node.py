"""Planning node: the core's services exposed to the cell.

Holds loaded parts in memory and answers three services:

* ``~/load_part``    read a CAD file and measure it;
* ``~/detect_seams`` recognise the seams;
* ``~/plan_job``     procedures, placement, paths and simulation.

It also publishes the result for anything that wants to watch:

* ``~/plan``          the latest :class:`weldforge_msgs/WeldPlan`, latched;
* ``~/seam_markers``  an RViz ``MarkerArray`` of the seams, coloured by joint
  type, with the torch direction drawn at every station -- which is how an
  operator checks the plan before anything moves.

Planning is **not** done in the service callback. A 200-seam weldment takes
tens of seconds, and a service callback that long blocks the executor and
makes the node look dead to everything else. Instead the request starts a
worker thread and the service returns as soon as the plan is ready, with the
node's own callback group left free so ``/plan``, markers and parameter
queries keep answering throughout.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

try:
    import builtin_interfaces.msg as builtin_interfaces
    import geometry_msgs.msg as geometry_msgs
    import rclpy
    import trajectory_msgs.msg as trajectory_msgs
    import weldforge_msgs.msg as wf_msgs
    import weldforge_msgs.srv as wf_srv
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError as exc:  # pragma: no cover - only importable inside ROS
    raise ImportError("weldforge_cell nodes need a sourced ROS 2 environment") from exc

from weldforge.cad import LoadOptions, backend_name, load_part
from weldforge.fixture import cell_robot, load_cell
from weldforge.model import Part
from weldforge.plan import PlanOptions, plan_from_part
from weldforge.seams import DetectOptions, detect_seams
from weldforge.units import m_to_mm, mm_to_m

from .conversions import job_to_plan_msg, pose_to_msg, seam_to_msg

# Joint-type colours, chosen to stay distinguishable in RViz's default dark
# background and for the common red-green colour deficiency.
_JOINT_COLOURS = {
    "fillet": (0.20, 0.70, 1.00),
    "tee": (0.20, 0.70, 1.00),
    "lap": (0.95, 0.75, 0.10),
    "butt": (0.40, 0.95, 0.40),
    "corner": (0.95, 0.45, 0.85),
    "edge": (0.70, 0.70, 0.70),
    "plug": (1.00, 0.55, 0.20),
    "unknown": (0.55, 0.55, 0.55),
}


class PlannerNode(Node):
    """Exposes the planning core as ROS services."""

    def __init__(self) -> None:
        super().__init__("weldforge_planner")

        self.declare_parameter("cell", "bench_cell")
        self.declare_parameter("base_material", "S355")
        self.declare_parameter("frame_id", "cell")
        self.declare_parameter("edge_spacing_mm", 2.0)
        self.declare_parameter("mesh_deflection_mm", 0.5)
        self.declare_parameter("min_seam_length_mm", 5.0)
        self.declare_parameter("min_confidence", 0.25)
        self.declare_parameter("leg_thickness_ratio", 0.7)
        self.declare_parameter("placement_grid_steps", 5)
        self.declare_parameter("placement_yaw_steps", 8)
        self.declare_parameter("placement_orientations", 12)
        self.declare_parameter("distortion_weight", 150.0)
        self.declare_parameter(
            "joint_names",
            ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        )
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("marker_torch_length_mm", 40.0)

        self._parts: dict[str, Part] = {}
        self._seams: dict[str, list] = {}
        self._lock = threading.RLock()
        self._last_plan: wf_msgs.WeldPlan | None = None

        # Services share a reentrant group so a long plan does not block the
        # others; the publishers keep their own group.
        services = ReentrantCallbackGroup()
        timers = MutuallyExclusiveCallbackGroup()

        self._srv_load = self.create_service(
            wf_srv.LoadPart, "~/load_part", self._on_load_part, callback_group=services
        )
        self._srv_detect = self.create_service(
            wf_srv.DetectSeams,
            "~/detect_seams",
            self._on_detect_seams,
            callback_group=services,
        )
        self._srv_plan = self.create_service(
            wf_srv.PlanJob, "~/plan_job", self._on_plan_job, callback_group=services
        )

        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pub_plan = self.create_publisher(wf_msgs.WeldPlan, "~/plan", latched)
        self._pub_markers = self.create_publisher(MarkerArray, "~/seam_markers", latched)
        self._timer = self.create_timer(2.0, self._republish, callback_group=timers)

        backend = backend_name()
        self.get_logger().info(
            f"WeldForge planner ready. CAD backend: {backend or 'NONE (no B-rep support)'}"
        )
        if backend is None:
            self.get_logger().warning(
                "No OpenCASCADE binding found: STEP and IGES files cannot be read. "
                "Install pythonocc-core (conda) or cadquery-ocp (pip) in this image."
            )

    # -- parameters ----------------------------------------------------------

    def _param(self, name: str):
        return self.get_parameter(name).value

    # -- services ------------------------------------------------------------

    def _on_load_part(self, request, response):
        path = Path(request.path)
        if not path.exists():
            response.success = False
            response.message = f"no such file: {path}"
            return response

        options = LoadOptions(
            edge_spacing_mm=(
                m_to_mm(request.edge_spacing)
                if request.edge_spacing > 0.0
                else float(self._param("edge_spacing_mm"))
            ),
            mesh_deflection_mm=(
                m_to_mm(request.mesh_deflection)
                if request.mesh_deflection > 0.0
                else float(self._param("mesh_deflection_mm"))
            ),
        )
        try:
            part = load_part(path, options, request.name or None)
        except Exception as exc:
            self.get_logger().error(f"load_part failed: {exc}")
            response.success = False
            response.message = f"{type(exc).__name__}: {exc}"
            return response

        with self._lock:
            self._parts[part.id] = part
            self._seams.pop(part.id, None)

        lo, hi = part.bbox
        response.success = True
        response.message = f"loaded {part.name}"
        response.part_id = part.id
        response.n_solids = len(part.solids)
        response.n_faces = len(part.faces)
        response.n_edges = len(part.edges)
        response.mass = float(part.mass_kg)
        response.bbox_min = geometry_msgs.Vector3(
            x=mm_to_m(float(lo[0])), y=mm_to_m(float(lo[1])), z=mm_to_m(float(lo[2]))
        )
        response.bbox_max = geometry_msgs.Vector3(
            x=mm_to_m(float(hi[0])), y=mm_to_m(float(hi[1])), z=mm_to_m(float(hi[2]))
        )
        response.thicknesses = [
            mm_to_m(t)
            for t in sorted({round(s.thickness_mm, 2) for s in part.solids if s.thickness_mm > 0})
        ]
        self.get_logger().info(
            f"loaded {part.name}: {len(part.solids)} solids, {len(part.faces)} faces, "
            f"{part.mass_kg:.1f} kg"
        )
        return response

    def _on_detect_seams(self, request, response):
        part = self._get_part(request.part_id)
        if part is None:
            response.success = False
            response.message = f"no part {request.part_id}; call load_part first"
            return response

        options = DetectOptions()
        options.min_length_mm = (
            m_to_mm(request.min_length)
            if request.min_length > 0.0
            else float(self._param("min_seam_length_mm"))
        )
        options.min_confidence = (
            float(request.min_confidence)
            if request.min_confidence > 0.0
            else float(self._param("min_confidence"))
        )

        try:
            seams, report = detect_seams(part, options)
        except Exception as exc:
            self.get_logger().error(f"detect_seams failed: {exc}")
            response.success = False
            response.message = f"{type(exc).__name__}: {exc}"
            return response

        with self._lock:
            self._seams[part.id] = seams

        frame = str(self._param("frame_id"))
        stamp = self.get_clock().now().to_msg()
        response.success = True
        response.message = f"{len(seams)} seams"
        response.seams = [
            seam_to_msg(s, wf_msgs, geometry_msgs, request.include_geometry, frame, stamp)
            for s in seams
        ]
        response.total_length = mm_to_m(sum(s.length_mm for s in seams))
        response.seconds = float(report.seconds)

        if bool(self._param("publish_markers")):
            self._publish_markers(seams, np.eye(4))
        self.get_logger().info(
            f"detected {len(seams)} seams, {sum(s.length_mm for s in seams):.0f} mm "
            f"in {report.seconds:.2f}s"
        )
        return response

    def _on_plan_job(self, request, response):
        part = self._get_part(request.part_id)
        if part is None:
            response.success = False
            response.message = f"no part {request.part_id}; call load_part first"
            return response

        cell_name = request.cell or str(self._param("cell"))
        try:
            cell = load_cell(cell_name)
        except FileNotFoundError as exc:
            response.success = False
            response.message = str(exc)
            return response

        options = PlanOptions()
        options.procedure.base_material = request.base_material or str(self._param("base_material"))
        options.procedure.leg_thickness_ratio = (
            float(request.leg_ratio)
            if request.leg_ratio > 0.0
            else float(self._param("leg_thickness_ratio"))
        )
        if request.wire_diameter > 0.0:
            options.procedure.wire_diameter_mm = m_to_mm(request.wire_diameter)
        options.do_placement = request.do_placement
        options.do_paths = request.do_paths
        options.do_simulate = request.do_simulate
        options.max_seams = request.max_seams or None
        options.min_seam_length_mm = float(self._param("min_seam_length_mm"))
        options.placement.grid_steps = int(self._param("placement_grid_steps"))
        options.placement.yaw_steps = int(self._param("placement_yaw_steps"))
        options.placement.max_orientations = int(self._param("placement_orientations"))
        options.path.sequence.distortion_weight = float(self._param("distortion_weight"))

        robot = cell_robot(cell)
        try:
            job = plan_from_part(part, cell, options, robot)
        except Exception as exc:
            self.get_logger().error(f"plan_job failed: {exc}")
            response.success = False
            response.message = f"{type(exc).__name__}: {exc}"
            return response

        if request.seam_ids:
            wanted = set(request.seam_ids)
            for seam in job.seams:
                seam.selected = seam.id in wanted

        joint_names = self._joint_names(robot)
        plan = job_to_plan_msg(
            job,
            wf_msgs,
            geometry_msgs,
            trajectory_msgs,
            builtin_interfaces,
            joint_names,
            frame_id=str(self._param("frame_id")),
            stamp=self.get_clock().now().to_msg(),
        )

        response.success = True
        response.message = "plan ready" if job.report.ok else "plan has problems; see report"
        response.plan = plan

        with self._lock:
            self._last_plan = plan
        self._pub_plan.publish(plan)
        if bool(self._param("publish_markers")):
            placement = job.placement.pose if job.placement is not None else np.eye(4)
            self._publish_markers(job.selected_seams, placement)

        report = job.report
        self.get_logger().info(
            f"planned {job.name}: {len(job.seams)} seams, {len(job.paths)} passes, "
            f"cycle {report.cycle_time_s:.1f}s, "
            f"{'OK' if report.ok else 'PROBLEMS: ' + '; '.join(report.errors[:3])}"
        )
        return response

    # -- helpers -------------------------------------------------------------

    def _get_part(self, part_id: str) -> Part | None:
        with self._lock:
            if part_id:
                return self._parts.get(part_id)
            # Empty id means "the most recently loaded part", which is what an
            # operator driving this from the command line expects.
            return next(reversed(self._parts.values()), None) if self._parts else None

    def _joint_names(self, robot) -> list[str]:
        declared = list(self._param("joint_names") or [])
        if len(declared) == robot.n_joints:
            return declared
        self.get_logger().warning(
            f"joint_names has {len(declared)} entries but the robot has "
            f"{robot.n_joints} joints; using the robot's own axis names"
        )
        return [j.name for j in robot.joints]

    def _republish(self) -> None:
        """Re-send the latched plan.

        Transient-local QoS already serves late joiners, but RViz is often
        started after the plan and occasionally misses the latched sample; a
        cheap periodic re-publish makes the display reliable.
        """
        with self._lock:
            plan = self._last_plan
        if plan is not None:
            self._pub_plan.publish(plan)

    # -- markers -------------------------------------------------------------

    def _publish_markers(self, seams, placement: np.ndarray) -> None:
        from weldforge.geom import transform as tf

        frame = str(self._param("frame_id"))
        stamp = self.get_clock().now().to_msg()
        torch_length = float(self._param("marker_torch_length_mm"))
        array = MarkerArray()

        # A full clear first: a new plan with fewer seams must not leave the
        # previous one's markers on screen.
        clear = Marker()
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        for index, seam in enumerate(seams):
            moved = seam.transformed(placement)
            colour = _JOINT_COLOURS.get(seam.joint_type.value, (0.6, 0.6, 0.6))

            line = Marker()
            line.header.frame_id = frame
            line.header.stamp = stamp
            line.ns = "weld_seams"
            line.id = index
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.004
            line.color = ColorRGBA(r=colour[0], g=colour[1], b=colour[2], a=1.0)
            line.pose.orientation.w = 1.0
            for point in moved.points:
                line.points.append(
                    geometry_msgs.Point(
                        x=mm_to_m(float(point[0])),
                        y=mm_to_m(float(point[1])),
                        z=mm_to_m(float(point[2])),
                    )
                )
            array.markers.append(line)

            # Wire direction at every station: this is what reveals a torch
            # pointing the wrong way, which a line alone cannot show.
            arrows = Marker()
            arrows.header.frame_id = frame
            arrows.header.stamp = stamp
            arrows.ns = "weld_wire_direction"
            arrows.id = index
            arrows.type = Marker.LINE_LIST
            arrows.action = Marker.ADD
            arrows.scale.x = 0.0015
            arrows.color = ColorRGBA(r=1.0, g=0.85, b=0.2, a=0.9)
            arrows.pose.orientation.w = 1.0
            step = max(1, moved.n_points // 20)
            for i in range(0, moved.n_points, step):
                start = moved.points[i]
                wire = moved.bisector[i] if len(moved.bisector) > i else np.array([0.0, 0.0, -1.0])
                tail = start - wire * torch_length
                for point in (tail, start):
                    arrows.points.append(
                        geometry_msgs.Point(
                            x=mm_to_m(float(point[0])),
                            y=mm_to_m(float(point[1])),
                            z=mm_to_m(float(point[2])),
                        )
                    )
            array.markers.append(arrows)

            label = Marker()
            label.header.frame_id = frame
            label.header.stamp = stamp
            label.ns = "weld_labels"
            label.id = index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.scale.z = 0.012
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
            label.pose = pose_to_msg(tf.pose(np.eye(3), moved.centroid), geometry_msgs)
            label.text = seam.label or f"{seam.joint_type.value} {seam.length_mm:.0f}mm"
            array.markers.append(label)

        self._pub_markers.publish(array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlannerNode()
    # Multi-threaded so a long plan cannot stall the rest of the node.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
