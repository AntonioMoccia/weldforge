#!/usr/bin/env bash
# Build the image and run every check that can only run inside it.
#
# This is the script that proves the ROS workspace is sound: colcon builds it,
# the message package generates its interfaces, and the node modules import
# against the real rclpy. None of that can be verified on a machine without
# ROS installed, which is the whole reason this file exists -- the core test
# suite on the development machine covers everything else.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> building the image"
docker compose -f docker/docker-compose.yml build dev

echo "==> colcon build"
docker compose -f docker/docker-compose.yml run --rm --no-deps dev bash -lc '
  set -e
  source /opt/ros/jazzy/setup.bash
  cd /ws && colcon build --symlink-install
'

echo "==> importing the nodes against the real rclpy"
docker compose -f docker/docker-compose.yml run --rm --no-deps dev bash -lc '
  set -e
  source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash
  python3 -c "
import weldforge_msgs.msg as m
from weldforge_cell import planner_node, welder_io_node, execution_node
print(\"interfaces:\", sorted(n for n in dir(m) if n[0].isupper()))
print(\"nodes import OK\")
"
'

echo "==> core test suite"
docker compose -f docker/docker-compose.yml run --rm --no-deps dev bash -lc '
  cd /opt/weldforge && python3 -m pytest -q
'

echo "==> colcon test"
docker compose -f docker/docker-compose.yml run --rm --no-deps dev bash -lc '
  source /opt/ros/jazzy/setup.bash
  cd /ws && colcon test --event-handlers console_direct+ && colcon test-result --verbose
'

echo "==> all checks passed"
