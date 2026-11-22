"""Weld job execution: the action server that actually welds.

Takes a :class:`weldforge_msgs/WeldPlan` and runs it, pass by pass:

1. write the pass's setpoints to the power source;
2. move to the approach point with the arc off;
3. strike the arc and wait for the machine to confirm current flow;
4. follow the welding trajectory at the planned travel speed;
5. crater-fill, extinguish, retract;
6. repeat, watching for arc faults throughout.

Why this is an action and not a service
---------------------------------------
Welding takes minutes, has to report progress, and must be cancellable. More
importantly it can **fail halfway**, with an arc burning and a robot moving,
and the recovery from that is specific: extinguish first, then retract, then
report. A service call cannot express any of that.

Arc sequencing is the part that must be right
---------------------------------------------
The arc and the motion are coupled, and getting the order wrong either burns a
hole at the start or leaves an unfused crater at the end:

* the arc is struck **at the start station, stationary**, and motion begins
  only once the machine confirms current -- striking while moving leaves a cold
  lap at the start of the bead;
* at the end, motion stops before the arc does, for the crater-fill time, so
  the pool fills the crater instead of leaving a pit;
* an arc that drops out mid-bead stops motion immediately: continuing to move
  with no arc deposits nothing and the seam has to be ground out and redone.

Trajectory execution is delegated to a ``FollowJointTrajectory`` controller --
``joint_trajectory_controller`` under ros2_control, or MoveIt's own. This node
never commands joints directly: the controller owns the arm's limits and
safety, and bypassing it would bypass those.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

try:
    import rclpy
    import weldforge_msgs.action as wf_action
    import weldforge_msgs.msg as wf_msgs
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Bool, Float64, String
except ImportError as exc:  # pragma: no cover
    raise ImportError("weldforge_cell nodes need a sourced ROS 2 environment") from exc

__all__ = ["ExecutionNode", "main"]


@dataclass
class ArcState:
    """Latest state reported by the welder I/O node."""

    established: bool = False
    current: float = 0.0
    voltage: float = 0.0
    fault: str = ""
    updated: float = 0.0

    @property
    def stale(self) -> bool:
        """True if the welder node has gone quiet.

        Treated as a fault while welding: an arc whose state is unknown is not
        an arc that may be left burning.
        """
        return self.updated > 0.0 and (time.time() - self.updated) > 1.0


class ExecutionNode(Node):
    """``ExecuteWeldJob`` action server."""

    def __init__(self) -> None:
        super().__init__("weldforge_execution")

        self.declare_parameter(
            "controller_action", "/joint_trajectory_controller/follow_joint_trajectory"
        )
        self.declare_parameter("welder_namespace", "/weldforge_welder_io")
        self.declare_parameter("trajectory_timeout_s", 300.0)
        self.declare_parameter("arc_strike_timeout_s", 3.0)
        self.declare_parameter("require_arc_feedback", True)
        self.declare_parameter("dry_run_speed_scale", 0.25)

        self._arc = ArcState()
        self._lock = threading.RLock()
        self._abort_reason = ""

        group = ReentrantCallbackGroup()
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        welder = str(self.get_parameter("welder_namespace").value).rstrip("/")

        self._pub_arc = self.create_publisher(Bool, f"{welder}/arc_command", reliable)
        self._pub_setpoints = self.create_publisher(
            wf_msgs.MachineSetpoints, f"{welder}/setpoints", reliable
        )
        self.create_subscription(
            Bool, f"{welder}/arc_established", self._on_arc, reliable, callback_group=group
        )
        self.create_subscription(
            Float64, f"{welder}/current", self._on_current, reliable, callback_group=group
        )
        self.create_subscription(
            Float64, f"{welder}/voltage", self._on_voltage, reliable, callback_group=group
        )
        self.create_subscription(
            String, f"{welder}/fault", self._on_fault, reliable, callback_group=group
        )

        self._controller = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("controller_action").value),
            callback_group=group,
        )
        self._server = ActionServer(
            self,
            wf_action.ExecuteWeldJob,
            "~/execute",
            execute_callback=self._execute,
            goal_callback=self._accept_goal,
            cancel_callback=self._accept_cancel,
            callback_group=group,
        )
        self.get_logger().info("weld job execution ready")

    # -- welder feedback -----------------------------------------------------

    def _on_arc(self, msg: Bool) -> None:
        with self._lock:
            self._arc.established = bool(msg.data)
            self._arc.updated = time.time()

    def _on_current(self, msg: Float64) -> None:
        with self._lock:
            self._arc.current = float(msg.data)
            self._arc.updated = time.time()

    def _on_voltage(self, msg: Float64) -> None:
        with self._lock:
            self._arc.voltage = float(msg.data)
            self._arc.updated = time.time()

    def _on_fault(self, msg: String) -> None:
        with self._lock:
            self._arc.fault = str(msg.data)
        self.get_logger().error(f"welder fault: {msg.data}")

    # -- goal handling -------------------------------------------------------

    def _accept_goal(self, goal) -> GoalResponse:
        plan = goal.plan
        if not plan.trajectories:
            self.get_logger().error("rejecting goal: the plan has no trajectories")
            return GoalResponse.REJECT
        if not plan.report.ok and not goal.dry_run:
            # A plan the planner itself flagged must not be welded. A dry run
            # is still allowed, because that is how an operator investigates.
            self.get_logger().error(
                "rejecting goal: the planner reported problems with this plan "
                f"({'; '.join(plan.report.errors[:3])}). Run it dry first."
            )
            return GoalResponse.REJECT
        if not plan.placement_feasible and not goal.dry_run:
            self.get_logger().error("rejecting goal: the part placement was not feasible")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _accept_cancel(self, goal) -> CancelResponse:
        self.get_logger().warning("cancel requested: extinguishing the arc")
        self._set_arc(False)
        return CancelResponse.ACCEPT

    # -- execution -----------------------------------------------------------

    def _execute(self, goal_handle):
        goal = goal_handle.request
        plan = goal.plan
        result = wf_action.ExecuteWeldJob.Result()
        result.passes_total = len(plan.trajectories)
        faults: list[str] = []
        arc_time = 0.0
        wire_used = 0.0

        scale = (
            float(self.get_parameter("dry_run_speed_scale").value)
            if goal.dry_run
            else max(0.01, min(1.0, goal.speed_scale or 1.0))
        )
        self.get_logger().info(
            f"executing {plan.job_name}: {result.passes_total} passes, "
            f"{'DRY RUN' if goal.dry_run else 'WELDING'}, speed scale {scale:.2f}"
        )

        if not self._controller.wait_for_server(timeout_sec=10.0):
            return self._fail(
                goal_handle,
                result,
                "the trajectory controller did not come up; is ros2_control running?",
            )

        procedures = {p.seam_id: p for p in plan.procedures}

        for index, trajectory in enumerate(plan.trajectories):
            if goal_handle.is_cancel_requested:
                self._set_arc(False)
                goal_handle.canceled()
                result.message = "cancelled"
                result.passes_completed = index
                return result

            seam_id = plan.trajectory_seam_ids[index]
            pass_index = int(plan.trajectory_pass_indices[index])
            procedure = procedures.get(seam_id)
            weld_pass = (
                procedure.passes[pass_index]
                if procedure is not None and pass_index < len(procedure.passes)
                else None
            )
            arc_flags = self._arc_flags(plan, index, len(trajectory.points))

            self._feedback(goal_handle, "moving", seam_id, pass_index, index, result.passes_total)

            # --- 1. setpoints ------------------------------------------------
            if weld_pass is not None and not goal.dry_run:
                self._pub_setpoints.publish(self._setpoints_for(weld_pass, pass_index))
                time.sleep(0.1)

            # --- 2/3. strike at the start station ----------------------------
            welding = any(arc_flags) and not goal.dry_run
            if welding:
                self._feedback(
                    goal_handle, "striking", seam_id, pass_index, index, result.passes_total
                )
                if not self._strike():
                    faults.append(f"{seam_id} pass {pass_index + 1}: arc failed to strike")
                    if goal.stop_on_arc_fault:
                        return self._fail(goal_handle, result, faults[-1], faults, index)
                    continue

            # --- 4. follow the trajectory ------------------------------------
            self._feedback(
                goal_handle,
                "welding" if welding else "moving",
                seam_id,
                pass_index,
                index,
                result.passes_total,
            )
            started = time.time()
            ok, message = self._follow(trajectory, scale, welding, goal_handle)
            elapsed = time.time() - started

            # --- 5. crater fill and extinguish -------------------------------
            if welding:
                if weld_pass is not None:
                    time.sleep(float(weld_pass.crater_fill))
                self._feedback(
                    goal_handle,
                    "retracting",
                    seam_id,
                    pass_index,
                    index,
                    result.passes_total,
                )
                self._set_arc(False)
                arc_time += elapsed
                if weld_pass is not None:
                    # Wire consumed = feed speed x time x section x density.
                    import math

                    radius = max(weld_pass.wire_diameter, 1e-6) / 2.0
                    volume = math.pi * radius * radius * weld_pass.wire_speed * elapsed
                    wire_used += volume * 7850.0  # kg, mild steel

            if not ok:
                faults.append(f"{seam_id} pass {pass_index + 1}: {message}")
                if goal.stop_on_arc_fault:
                    return self._fail(goal_handle, result, faults[-1], faults, index)

            result.passes_completed = index + 1

        self._set_arc(False)
        result.success = not faults
        result.message = "finished" if not faults else f"{len(faults)} pass(es) faulted"
        result.arc_time = arc_time
        result.wire_used = wire_used
        result.faults = faults
        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        self.get_logger().info(
            f"execution {'finished' if result.success else 'aborted'}: "
            f"{result.passes_completed}/{result.passes_total} passes, "
            f"{arc_time:.1f} s arc, {wire_used * 1000:.0f} g wire"
        )
        return result

    # -- steps ---------------------------------------------------------------

    def _strike(self) -> bool:
        """Strike the arc and wait for the machine to confirm current flow."""
        self._set_arc(True)
        if not bool(self.get_parameter("require_arc_feedback").value):
            time.sleep(0.4)
            return True

        deadline = time.time() + float(self.get_parameter("arc_strike_timeout_s").value)
        while time.time() < deadline:
            with self._lock:
                if self._arc.established:
                    return True
                if self._arc.fault:
                    return False
            time.sleep(0.02)
        self._set_arc(False)
        return False

    def _follow(self, trajectory, scale: float, welding: bool, goal_handle):
        """Send a trajectory to the controller and wait, watching the arc."""
        if scale != 1.0:
            trajectory = self._scale(trajectory, scale)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        future = self._controller.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None:
            return False, "the controller did not accept the trajectory"
        handle = future.result()
        if not handle.accepted:
            return False, "the controller rejected the trajectory"

        result_future = handle.get_result_async()
        timeout = float(self.get_parameter("trajectory_timeout_s").value)
        deadline = time.time() + timeout

        while not result_future.done():
            if time.time() > deadline:
                handle.cancel_goal_async()
                return False, f"the trajectory did not finish within {timeout:.0f} s"
            if goal_handle.is_cancel_requested:
                handle.cancel_goal_async()
                return False, "cancelled"
            if welding:
                with self._lock:
                    lost = not self._arc.established
                    stale = self._arc.stale
                    fault = self._arc.fault
                if fault:
                    handle.cancel_goal_async()
                    return False, f"welder fault: {fault}"
                if lost:
                    # Keep moving with no arc and the seam has to be ground
                    # out: stop at once.
                    handle.cancel_goal_async()
                    return False, "the arc went out during the weld"
                if stale:
                    handle.cancel_goal_async()
                    return False, "lost contact with the welder node"
            rclpy.spin_once(self, timeout_sec=0.02)

        outcome = result_future.result()
        if outcome is None:
            return False, "the controller returned no result"
        code = outcome.result.error_code
        if code != FollowJointTrajectory.Result.SUCCESSFUL:
            return False, f"the controller reported error code {code}"
        return True, "ok"

    # -- helpers -------------------------------------------------------------

    def _arc_flags(self, plan, index: int, n_points: int) -> list[bool]:
        """Unpack this trajectory's slice of the flattened arc-on array."""
        if not plan.arc_on_offsets or index >= len(plan.arc_on_offsets):
            return [False] * n_points
        start = int(plan.arc_on_offsets[index])
        end = (
            int(plan.arc_on_offsets[index + 1])
            if index + 1 < len(plan.arc_on_offsets)
            else len(plan.arc_on)
        )
        return [bool(v) for v in plan.arc_on[start:end]]

    def _setpoints_for(self, weld_pass, pass_index: int) -> wf_msgs.MachineSetpoints:
        out = wf_msgs.MachineSetpoints()
        out.pass_index = pass_index
        out.current = float(weld_pass.current)
        out.voltage = float(weld_pass.voltage)
        out.wire_speed = float(weld_pass.wire_speed)
        out.gas_flow = float(weld_pass.gas_flow)
        return out

    def _scale(self, trajectory, scale: float):
        """Stretch a trajectory's timestamps; positions are untouched."""
        import copy

        from builtin_interfaces.msg import Duration

        out = copy.deepcopy(trajectory)
        for point in out.points:
            total = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            total /= scale
            seconds = int(total)
            point.time_from_start = Duration(
                sec=seconds, nanosec=int(round((total - seconds) * 1e9))
            )
        return out

    def _set_arc(self, state: bool) -> None:
        self._pub_arc.publish(Bool(data=state))

    def _feedback(
        self,
        goal_handle,
        stage: str,
        seam_id: str,
        pass_index: int,
        completed: int,
        total: int,
    ) -> None:
        feedback = wf_action.ExecuteWeldJob.Feedback()
        feedback.stage = stage
        feedback.seam_id = seam_id
        feedback.pass_index = pass_index
        feedback.passes_completed = completed
        feedback.passes_total = total
        feedback.progress = (completed / total) if total else 0.0
        with self._lock:
            feedback.current = self._arc.current
            feedback.voltage = self._arc.voltage
            feedback.arc_established = self._arc.established
        goal_handle.publish_feedback(feedback)

    def _fail(self, goal_handle, result, message: str, faults=None, completed: int = 0):
        """Extinguish the arc, abort the goal and report."""
        self._set_arc(False)
        result.success = False
        result.message = message
        result.faults = faults if faults is not None else [message]
        result.passes_completed = completed
        goal_handle.abort()
        self.get_logger().error(f"execution aborted: {message}")
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExecutionNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        node._set_arc(False)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
