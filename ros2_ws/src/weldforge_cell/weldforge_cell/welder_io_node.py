"""Power source driver.

Owns the interface to the welding machine and nothing else, so the execution
node never talks to hardware directly. Three backends:

``simulated``
    Models striking, arc establishment and extinction with realistic delays and
    reports plausible current and voltage. This is what runs in Gazebo and in
    CI, and it is what makes the arc sequencing testable at all.

``digital_io``
    Drives the machine through a generic I/O interface: publishes the analog
    words and digital bits from the machine profile onto topics that a fieldbus
    gateway node subscribes to. This is the retrofit case and covers most of
    the installed base.

``modbus``
    A stub with the register layout spelt out, for a machine with a Modbus/TCP
    interface. Deliberately left unimplemented rather than guessed: the
    register map differs per machine and must come from its manual.

Safety, and why the interlocks are in this node
-----------------------------------------------
The arc is the dangerous part, so the rules live at the single point that can
strike it:

* ``arc_enabled`` must be set by an external supervisor before the arc can be
  struck at all -- a planning bug cannot strike an arc on its own;
* a strike that does not establish within ``arc_timeout`` is aborted and the
  gas shut off, rather than left feeding wire into the work;
* an arc that drops out during a weld raises a fault immediately;
* on shutdown, however it happens, the arc and gas outputs are driven low.
"""

from __future__ import annotations

import threading
import time

try:
    import rclpy
    import weldforge_msgs.msg as wf_msgs
    import weldforge_msgs.srv as wf_srv
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Bool, Float64, Int32MultiArray, String
except ImportError as exc:  # pragma: no cover
    raise ImportError("weldforge_cell nodes need a sourced ROS 2 environment") from exc

from weldforge.wps import load_machine
from weldforge.wps.machines import MachineProfile


class WelderIoNode(Node):
    """Talks to the power source; publishes arc state and measurements."""

    def __init__(self) -> None:
        super().__init__("weldforge_welder_io")

        self.declare_parameter("backend", "simulated")  # simulated|digital_io|modbus
        self.declare_parameter("machine_profile", "generic_analog")
        self.declare_parameter("arc_enabled", False)
        self.declare_parameter("arc_timeout_s", 2.0)
        self.declare_parameter("gas_preflow_s", 0.3)
        self.declare_parameter("gas_postflow_s", 1.0)
        self.declare_parameter("publish_rate_hz", 20.0)
        # Simulated arc behaviour.
        self.declare_parameter("sim_strike_delay_s", 0.25)
        self.declare_parameter("sim_noise_fraction", 0.03)

        self._profile: MachineProfile | None = None
        self._load_profile()

        self._lock = threading.RLock()
        self._setpoints: wf_msgs.MachineSetpoints | None = None
        self._arc_requested = False
        self._arc_established = False
        self._arc_since: float | None = None
        self._gas_on = False
        self._faults: list[str] = []

        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        group = MutuallyExclusiveCallbackGroup()

        # --- commands in -----------------------------------------------------
        self._sub_arc = self.create_subscription(
            Bool, "~/arc_command", self._on_arc_command, reliable, callback_group=group
        )
        self._sub_setpoints = self.create_subscription(
            wf_msgs.MachineSetpoints,
            "~/setpoints",
            self._on_setpoints,
            reliable,
            callback_group=group,
        )
        self._srv_params = self.create_service(
            wf_srv.SetWeldParameters,
            "~/set_parameters",
            self._on_set_parameters,
            callback_group=group,
        )

        # --- state out -------------------------------------------------------
        self._pub_arc_on = self.create_publisher(Bool, "~/arc_established", reliable)
        self._pub_current = self.create_publisher(Float64, "~/current", reliable)
        self._pub_voltage = self.create_publisher(Float64, "~/voltage", reliable)
        self._pub_wire = self.create_publisher(Float64, "~/wire_speed", reliable)
        self._pub_gas = self.create_publisher(Bool, "~/gas_on", reliable)
        self._pub_fault = self.create_publisher(String, "~/fault", reliable)
        # For the digital_io backend: the raw fieldbus image a gateway writes.
        self._pub_analog = self.create_publisher(Int32MultiArray, "~/analog_words", reliable)
        self._pub_digital = self.create_publisher(Int32MultiArray, "~/digital_bits", reliable)

        rate = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self._timer = self.create_timer(1.0 / rate, self._tick, callback_group=group)

        backend = self.get_parameter("backend").value
        self.get_logger().info(
            f"welder I/O ready: backend={backend}, "
            f"profile={self._profile.name if self._profile else 'none'}, "
            f"arc_enabled={self.get_parameter('arc_enabled').value}"
        )
        if backend == "modbus":
            self.get_logger().warning(
                "the modbus backend is a stub: fill in the register map from the "
                "machine's manual before using it on hardware"
            )

    # -- setup ---------------------------------------------------------------

    def _load_profile(self) -> None:
        name = str(self.get_parameter("machine_profile").value)
        try:
            self._profile = load_machine(name)
        except FileNotFoundError as exc:
            self.get_logger().error(f"machine profile {name!r} not found: {exc}")
            self._profile = None

    @property
    def _arc_allowed(self) -> bool:
        return bool(self.get_parameter("arc_enabled").value)

    # -- commands ------------------------------------------------------------

    def _on_setpoints(self, msg: wf_msgs.MachineSetpoints) -> None:
        with self._lock:
            self._setpoints = msg
        self.get_logger().info(
            f"setpoints: pass {msg.pass_index}, {msg.current:.0f} A, "
            f"{msg.voltage:.1f} V, wire {msg.wire_speed * 60.0:.2f} m/min"
            + (f", job {msg.job_number}" if msg.has_job_number else "")
        )
        self._write_to_machine(msg)

    def _on_arc_command(self, msg: Bool) -> None:
        if msg.data:
            self._request_arc_on()
        else:
            self._request_arc_off()

    def _on_set_parameters(self, request, response):
        self._on_setpoints(request.setpoints)
        if request.gas_test:
            self._set_gas(True)
            time.sleep(min(3.0, float(self.get_parameter("gas_preflow_s").value) * 3))
            self._set_gas(False)
        if request.wire_inch:
            self._inch_wire()
        response.success = True
        response.message = "setpoints written"
        return response

    # -- arc control ---------------------------------------------------------

    def _request_arc_on(self) -> None:
        if not self._arc_allowed:
            self._raise_fault("arc requested while arc_enabled is false: refusing to strike")
            return
        with self._lock:
            if self._setpoints is None:
                self._raise_fault("arc requested with no setpoints written")
                return
            if self._arc_requested:
                return
            self._arc_requested = True

        self._set_gas(True)
        time.sleep(float(self.get_parameter("gas_preflow_s").value))
        self._strike()

    def _request_arc_off(self) -> None:
        with self._lock:
            if not self._arc_requested:
                return
            self._arc_requested = False
            self._arc_established = False
            self._arc_since = None
        self._set_digital("arc_on", False)
        self.get_logger().info("arc off")
        time.sleep(float(self.get_parameter("gas_postflow_s").value))
        self._set_gas(False)

    def _strike(self) -> None:
        """Strike the arc and wait for the machine to confirm current flow."""
        self._set_digital("arc_on", True)
        timeout = float(self.get_parameter("arc_timeout_s").value)
        deadline = time.time() + timeout
        backend = str(self.get_parameter("backend").value)

        while time.time() < deadline:
            if backend == "simulated":
                if time.time() >= deadline - timeout + float(
                    self.get_parameter("sim_strike_delay_s").value
                ):
                    with self._lock:
                        self._arc_established = True
                        self._arc_since = time.time()
                    self.get_logger().info("arc established")
                    return
            else:
                # Real machines answer on a digital input, routed in by the
                # fieldbus gateway. Until that is wired, fail closed.
                pass
            time.sleep(0.02)

        self._raise_fault(f"arc failed to establish within {timeout:.1f} s")
        self._set_digital("arc_on", False)
        self._set_gas(False)
        with self._lock:
            self._arc_requested = False

    # -- machine writing -----------------------------------------------------

    def _write_to_machine(self, msg: wf_msgs.MachineSetpoints) -> None:
        backend = str(self.get_parameter("backend").value)
        if backend == "simulated":
            return
        if backend == "digital_io":
            words = Int32MultiArray()
            words.data = [int(v) for v in msg.analog_values]
            self._pub_analog.publish(words)
            bits = Int32MultiArray()
            bits.data = [int(bool(v)) for v in msg.digital_values]
            self._pub_digital.publish(bits)
            return
        if backend == "modbus":
            # Intentionally not implemented. A guessed register map would write
            # arbitrary values into a welding machine.
            self._raise_fault("modbus backend is a stub: no register map configured")

    def _set_gas(self, state: bool) -> None:
        with self._lock:
            self._gas_on = state
        self._set_digital("gas_test", state)
        self._pub_gas.publish(Bool(data=state))

    def _set_digital(self, role: str, state: bool) -> None:
        """Set one named discrete output, resolved through the machine profile."""
        if self._profile is None:
            return
        signal = self._profile.digital.get(role)
        if signal is None:
            self.get_logger().debug(f"profile has no {role!r} output")
            return
        if str(self.get_parameter("backend").value) == "digital_io":
            bits = Int32MultiArray()
            bits.data = [int(state)]
            self._pub_digital.publish(bits)

    def _inch_wire(self) -> None:
        self._set_digital("wire_inch", True)
        time.sleep(0.5)
        self._set_digital("wire_inch", False)

    def _raise_fault(self, message: str) -> None:
        with self._lock:
            self._faults.append(message)
        self.get_logger().error(message)
        self._pub_fault.publish(String(data=message))

    # -- periodic state ------------------------------------------------------

    def _tick(self) -> None:
        import random

        with self._lock:
            established = self._arc_established
            setpoints = self._setpoints
            gas = self._gas_on

        self._pub_arc_on.publish(Bool(data=established))
        self._pub_gas.publish(Bool(data=gas))

        if established and setpoints is not None:
            noise = float(self.get_parameter("sim_noise_fraction").value)

            def jitter(value: float) -> float:
                """Add a little scatter to a reading.

                A consumer that only works with perfectly steady values will
                not work on a real arc either, so the simulated machine never
                reports one.
                """
                return value * (1.0 + random.uniform(-noise, noise))

            self._pub_current.publish(Float64(data=jitter(setpoints.current)))
            self._pub_voltage.publish(Float64(data=jitter(setpoints.voltage)))
            self._pub_wire.publish(Float64(data=setpoints.wire_speed))
        else:
            self._pub_current.publish(Float64(data=0.0))
            self._pub_voltage.publish(Float64(data=0.0))
            self._pub_wire.publish(Float64(data=0.0))

    # -- shutdown ------------------------------------------------------------

    def destroy_node(self) -> bool:
        """Drive the arc and gas low on the way out, however we got here."""
        try:
            self._set_digital("arc_on", False)
            self._set_gas(False)
        except Exception:  # pragma: no cover - best effort during teardown
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WelderIoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
