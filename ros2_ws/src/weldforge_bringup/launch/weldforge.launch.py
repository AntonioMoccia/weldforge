"""Bring up everything: simulation or hardware, MoveIt, and the WeldForge cell.

.. code-block:: bash

    # Everything simulated. This is the default and it cannot strike an arc.
    ros2 launch weldforge_bringup weldforge.launch.py

    # Real robot and real power source. The arc interlock still has to be
    # armed explicitly.
    ros2 launch weldforge_bringup weldforge.launch.py \
        use_sim:=false welder_backend:=digital_io arc_enabled:=true

Order matters: the robot's controllers must be up before MoveIt looks for
them, and the welder node must be up before the execution node subscribes to
its feedback. The delays below are crude but reliable; a production cell
should use lifecycle nodes instead.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    cell = LaunchConfiguration("cell")
    use_sim = LaunchConfiguration("use_sim")
    arc_enabled = LaunchConfiguration("arc_enabled")

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("weldforge_sim"), "launch", "sim.launch.py"])
        ),
        launch_arguments={"cell": cell}.items(),
        condition=IfCondition(use_sim),
    )

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("weldforge_moveit_config"),
                    "launch",
                    "moveit.launch.py",
                ]
            )
        ),
        launch_arguments={"cell": cell}.items(),
    )

    weldforge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("weldforge_cell"), "launch", "cell.launch.py"])
        ),
        launch_arguments={
            "cell": cell,
            "base_material": LaunchConfiguration("base_material"),
            "welder_backend": LaunchConfiguration("welder_backend"),
            "machine_profile": LaunchConfiguration("machine_profile"),
            "arc_enabled": arc_enabled,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("cell", default_value="bench_cell"),
            DeclareLaunchArgument("base_material", default_value="S355"),
            DeclareLaunchArgument("use_sim", default_value="true"),
            DeclareLaunchArgument("welder_backend", default_value="simulated"),
            DeclareLaunchArgument("machine_profile", default_value="generic_analog"),
            DeclareLaunchArgument("arc_enabled", default_value="false"),
            LogInfo(
                condition=UnlessCondition(use_sim),
                msg="*** HARDWARE MODE: the real robot will move. ***",
            ),
            sim,
            # MoveIt after the controllers exist, the cell after MoveIt.
            TimerAction(period=6.0, actions=[moveit]),
            TimerAction(period=10.0, actions=[weldforge]),
        ]
    )
