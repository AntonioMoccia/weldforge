"""Launch the planning node on its own.

Useful for working on a part without the rest of the cell: load a STEP file,
detect seams, plan, and look at the result in RViz.

.. code-block:: bash

    ros2 launch weldforge_cell planner.launch.py cell:=bench_cell
    ros2 service call /weldforge_planner/load_part weldforge_msgs/srv/LoadPart \
        "{path: /work/part.step}"
    ros2 service call /weldforge_planner/plan_job weldforge_msgs/srv/PlanJob \
        "{part_id: '', do_placement: true, do_paths: true, do_simulate: true}"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    cell = LaunchConfiguration("cell")
    material = LaunchConfiguration("base_material")
    params = PathJoinSubstitution([FindPackageShare("weldforge_cell"), "config", "planner.yaml"])

    return LaunchDescription(
        [
            DeclareLaunchArgument("cell", default_value="bench_cell"),
            DeclareLaunchArgument("base_material", default_value="S355"),
            Node(
                package="weldforge_cell",
                executable="planner_node",
                name="weldforge_planner",
                output="screen",
                parameters=[params, {"cell": cell, "base_material": material}],
            ),
        ]
    )
