"""Show a cell in RViz with sliders for the joints.

The quickest way to check that a cell's geometry is right: bench height, how
far the part sits from the robot, where the clamps are, and whether the torch
frame points the way it should.

.. code-block:: bash

    ros2 launch weldforge_description view_cell.launch.py cell:=bench_cell
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _nodes(context, *args, **kwargs):
    from pathlib import Path

    cell = LaunchConfiguration("cell").perform(context)
    gui = LaunchConfiguration("gui").perform(context).lower() in ("1", "true", "yes")
    share = Path(FindPackageShare("weldforge_description").perform(context))
    urdf = share / "urdf" / f"{cell}.urdf"
    if not urdf.exists():
        raise RuntimeError(
            f"{urdf} does not exist. Generate it with: weldforge urdf --cell {cell} "
            f"-o {share / 'urdf'}"
        )
    description = urdf.read_text()

    nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": description}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            arguments=["-d", str(share / "rviz" / "cell.rviz")],
        ),
    ]
    if gui:
        nodes.append(
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                output="screen",
            )
        )
    else:
        nodes.append(
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                output="screen",
            )
        )
    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("cell", default_value="bench_cell"),
            DeclareLaunchArgument("gui", default_value="true"),
            OpaqueFunction(function=_nodes),
        ]
    )
