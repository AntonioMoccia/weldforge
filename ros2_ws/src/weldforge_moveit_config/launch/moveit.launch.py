"""Bring up MoveIt 2 for a WeldForge cell.

.. code-block:: bash

    ros2 launch weldforge_moveit_config moveit.launch.py cell:=bench_cell

Loads the generated URDF and SRDF from ``weldforge_description``, both
planning pipelines (OMPL for air moves, Pilz for welding moves), and RViz with
the MoveIt panel.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _nodes(context, *args, **kwargs):
    from pathlib import Path

    import yaml

    cell = LaunchConfiguration("cell").perform(context)
    description_share = Path(FindPackageShare("weldforge_description").perform(context))
    config_share = Path(FindPackageShare("weldforge_moveit_config").perform(context))

    urdf = description_share / "urdf" / f"{cell}.urdf"
    srdf = description_share / "urdf" / f"{cell}.srdf"
    for path in (urdf, srdf):
        if not path.exists():
            raise RuntimeError(
                f"{path} does not exist. Generate it with: "
                f"weldforge urdf --cell {cell} -o {description_share / 'urdf'}"
            )

    def load(name: str) -> dict:
        return yaml.safe_load((config_share / "config" / name).read_text()) or {}

    robot_description = {"robot_description": urdf.read_text()}
    robot_description_semantic = {"robot_description_semantic": srdf.read_text()}
    kinematics = {"robot_description_kinematics": load("kinematics.yaml")}
    limits = {"robot_description_planning": load("joint_limits.yaml")}
    ompl = {"ompl": load("ompl_planning.yaml")}
    pilz = {"pilz_industrial_motion_planner": load("pilz_industrial_motion_planner.yaml")}
    controllers = load("moveit_controllers.yaml")

    move_group_params = [
        robot_description,
        robot_description_semantic,
        kinematics,
        limits,
        ompl,
        pilz,
        controllers,
        {
            # Both pipelines are loaded; a request names the one it wants.
            "planning_pipelines": ["ompl", "pilz_industrial_motion_planner"],
            "default_planning_pipeline": "pilz_industrial_motion_planner",
            "publish_robot_description_semantic": True,
            "publish_planning_scene": True,
            "publish_geometry_updates": True,
            "publish_state_updates": True,
            "publish_transforms_updates": True,
        },
    ]

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[robot_description],
        ),
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=move_group_params,
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            arguments=["-d", str(description_share / "rviz" / "cell.rviz")],
            parameters=[
                robot_description,
                robot_description_semantic,
                kinematics,
                limits,
            ],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("cell", default_value="bench_cell"),
            OpaqueFunction(function=_nodes),
        ]
    )
