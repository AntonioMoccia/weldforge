# WeldForge development and cell image.
#
# ROS 2 Jazzy on Ubuntu 24.04, plus the planning core and an OpenCASCADE
# binding. The binding is the only awkward dependency: pythonocc-core is
# conda-only, so this image uses the pip wheels (cadquery-ocp) instead, which
# keeps everything installable with apt and pip and avoids dragging conda into
# a robot cell.
FROM ros:jazzy-ros-base

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# --- system packages -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip python3-venv python3-numpy python3-scipy python3-yaml \
      build-essential cmake git \
      libgl1 libglu1-mesa libxrender1 libxcursor1 libxinerama1 libfontconfig1 \
      ros-jazzy-moveit \
      ros-jazzy-moveit-planners-ompl \
      ros-jazzy-pilz-industrial-motion-planner \
      ros-jazzy-ros2-control \
      ros-jazzy-ros2-controllers \
      ros-jazzy-joint-trajectory-controller \
      ros-jazzy-joint-state-broadcaster \
      ros-jazzy-joint-state-publisher-gui \
      ros-jazzy-robot-state-publisher \
      ros-jazzy-controller-manager \
      ros-jazzy-rviz2 \
      ros-jazzy-ros-gz-sim \
      ros-jazzy-ros-gz-bridge \
      ros-jazzy-gz-ros2-control \
      ros-jazzy-control-msgs \
      ros-jazzy-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

# --- the planning core -----------------------------------------------------
# Jazzy's Python is externally managed, so pip needs the override. This is an
# application image, not a shared system, so that is the right call.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY pyproject.toml README.md /opt/weldforge/
COPY src /opt/weldforge/src
COPY data /opt/weldforge/data

RUN pip install --no-cache-dir -e "/opt/weldforge[api,sim]" \
 && pip install --no-cache-dir "cadquery-ocp>=7.7" \
 && python3 -c "from weldforge.cad import backend_name; print('CAD backend:', backend_name())"

# --- the ROS workspace -----------------------------------------------------
WORKDIR /ws
COPY ros2_ws/src /ws/src

RUN source /opt/ros/jazzy/setup.bash \
 && rosdep update --rosdistro jazzy \
 && rosdep install --from-paths src --ignore-src -y --rosdistro jazzy \
      --skip-keys "python3-numpy python3-scipy python3-yaml" \
 && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source both environments for every shell, so `ros2 launch` just works.
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc \
 && echo 'source /ws/install/setup.bash' >> /root/.bashrc \
 && echo 'export PYTHONPATH=/opt/weldforge/src:$PYTHONPATH' >> /root/.bashrc

ENV WELDFORGE_DATA=/opt/weldforge/data

CMD ["bash", "-lc", "ros2 launch weldforge_bringup weldforge.launch.py"]
