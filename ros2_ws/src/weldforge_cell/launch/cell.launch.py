"""Launch the whole cell: planner, welder I/O and execution.

.. code-block:: bash

    # Simulated power source, arc inhibited -- the safe default.
    ros2 launch weldforge_cell cell.launch.py

    # Real machine over a fieldbus gateway, arc armed by a human.
    ros2 launch weldforge_cell cell.launch.py \
        welder_backend:=digital_io machine_profile:=fronius_tps arc_enabled:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    share = FindPackageShare("weldforge_cell")
    planner_params = PathJoinSubstitution([share, "config", "planner.yaml"])
    welder_params = PathJoinSubstitution([share, "config", "welder.yaml"])
    execution_params = PathJoinSubstitution([share, "config", "execution.yaml"])

    arc_enabled = LaunchConfiguration("arc_enabled")

    return LaunchDescription(
        [
            DeclareLaunchArgument("cell", default_value="bench_cell"),
            DeclareLaunchArgument("base_material", default_value="S355"),
            DeclareLaunchArgument(
                "welder_backend",
                default_value="simulated",
                description="simulated | digital_io | modbus",
            ),
            DeclareLaunchArgument("machine_profile", default_value="generic_analog"),
            DeclareLaunchArgument(
                "arc_enabled",
                default_value="false",
                description="Master arc interlock. Must be true to weld.",
            ),
            LogInfo(
                condition=IfCondition(arc_enabled),
                msg=[
                    "*** ARC IS ARMED. The cell can strike a live arc. ",
                    "Confirm the fixture is clear and the guarding is closed. ***",
                ],
            ),
            Node(
                package="weldforge_cell",
                executable="planner_node",
                name="weldforge_planner",
                output="screen",
                parameters=[
                    planner_params,
                    {
                        "cell": LaunchConfiguration("cell"),
                        "base_material": LaunchConfiguration("base_material"),
                    },
                ],
            ),
            Node(
                package="weldforge_cell",
                executable="welder_io_node",
                name="weldforge_welder_io",
                output="screen",
                parameters=[
                    welder_params,
                    {
                        "backend": LaunchConfiguration("welder_backend"),
                        "machine_profile": LaunchConfiguration("machine_profile"),
                        "arc_enabled": arc_enabled,
                    },
                ],
            ),
            Node(
                package="weldforge_cell",
                executable="execution_node",
                name="weldforge_execution",
                output="screen",
                parameters=[execution_params],
            ),
        ]
    )
