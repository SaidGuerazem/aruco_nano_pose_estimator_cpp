cmake_minimum_required(VERSION 3.8)
project(aruco_nano_pose_estimator_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV 4.9 REQUIRED)

add_executable(aruco_nano_pose_estimator
  src/aruco_nano_pose_estimator.cpp
)

ament_target_dependencies(aruco_nano_pose_estimator
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  nav_msgs
  cv_bridge
)

target_include_directories(aruco_nano_pose_estimator PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(aruco_nano_pose_estimator
  ${OpenCV_LIBS}
)

install(TARGETS
  aruco_nano_pose_estimator
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
