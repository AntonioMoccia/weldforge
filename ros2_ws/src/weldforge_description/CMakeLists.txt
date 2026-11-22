cmake_minimum_required(VERSION 3.16)
project(weldforge_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf launch rviz config meshes
  DESTINATION share/${PROJECT_NAME}
  PATTERN ".gitkeep" EXCLUDE
)

ament_package()
