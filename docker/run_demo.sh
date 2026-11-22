#!/usr/bin/env bash
# Plan a part end to end inside the container, through the ROS services.
#
#   ./docker/run_demo.sh part.step [cell]
#
# The file must sit in ./work, which is mounted at /work in the container.
# Loading it through the planner node's services exercises exactly the path a
# CAD plugin takes, rather than a shortcut that only works in a test.
set -euo pipefail
cd "$(dirname "$0")/.."

PART="${1:?usage: run_demo.sh <file in ./work> [cell]}"
CELL="${2:-bench_cell}"
NAME="$(basename "$PART")"

if [[ ! -f "work/${NAME}" ]]; then
  echo "error: work/${NAME} not found. Copy the CAD file into ./work first." >&2
  exit 1
fi

docker compose -f docker/docker-compose.yml run --rm --no-deps dev bash -lc "
  set -e
  source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash

  ros2 launch weldforge_cell planner.launch.py cell:=${CELL} &
  LAUNCH=\$!
  trap 'kill \$LAUNCH 2>/dev/null || true' EXIT
  sleep 5

  echo '==> load'
  ros2 service call /weldforge_planner/load_part weldforge_msgs/srv/LoadPart \
    \"{path: '/work/${NAME}'}\"

  echo '==> detect'
  ros2 service call /weldforge_planner/detect_seams weldforge_msgs/srv/DetectSeams \
    \"{part_id: '', include_geometry: false}\"

  echo '==> plan'
  ros2 service call /weldforge_planner/plan_job weldforge_msgs/srv/PlanJob \
    \"{part_id: '', do_placement: true, do_paths: true, do_simulate: true}\"
"
