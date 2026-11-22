"""Gazebo simulation of the welding cell.

.. code-block:: bash

    ros2 launch weldforge_sim sim.launch.py cell:=bench_cell

Brings up Gazebo with the cell world, spawns the generated cell model,
ros2_control with the simulated hardware interface, and the bridge. The
welding arc is simulated by the welder node's ``simulated`` backend, not by
Gazebo -- see the note in ``worlds/welding_cell.sdf``.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _nodes(context, *args, **kwargs):
    from pathlib import Path

    cell = LaunchConfiguration("cell").perform(context)
    description_share = Path(FindPackageShare("weldforge_description").perform(context))
    sim_share = Path(FindPackageShare("weldforge_sim").perform(context))

    urdf = description_share / "urdf" / f"{cell}.urdf"
    if not urdf.exists():
        raise RuntimeError(
            f"{urdf} does not exist. Generate it with: "
            f"weldforge urdf --cell {cell} -o {description_share / 'urdf'}"
        )

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": urdf.read_text(), "use_sim_time": True}],
        ),
        Node(
            package="ros_gz_sim",
            executable="create",
            output="screen",
            arguments=["-name", "weldforge_cell", "-topic", "robot_description"],
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            output="screen",
            arguments=[
                "--ros-args",
                "-p",
                f"config_file:={sim_share / 'config' / 'gz_bridge.yaml'}",
            ],
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster"],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_trajectory_controller"],
            output="screen",
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    world = PathJoinSubstitution([FindPackageShare("weldforge_sim"), "worlds", "welding_cell.sdf"])
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])
        ),
        launch_arguments={"gz_args": ["-r -v3 ", world]}.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("cell", default_value="bench_cell"),
            gz,
            OpaqueFunction(function=_nodes),
        ]
    )
