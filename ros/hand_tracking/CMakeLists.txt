cmake_minimum_required(VERSION 2.8.3)
project(hand_tracking)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  bio_ik
  roscpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  message_filters
  std_msgs
  geometry_msgs
  sensor_msgs
  pr2_mechanism_msgs
  pr2_controllers_msgs
  actionlib
)
find_package(Eigen3 REQUIRED)

catkin_package()

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(arm_track_human_hand src/arm_track_human_hand.cpp)
add_dependencies(arm_track_human_hand ${catkin_EXPORTED_TARGETS})
target_link_libraries(arm_track_human_hand ${catkin_LIBRARIES} ccd)

add_executable(pr2_kinect_arm_track_human_hand src/pr2_kinect_arm_track_human_hand.cpp)
add_dependencies(pr2_kinect_arm_track_human_hand ${catkin_EXPORTED_TARGETS})
target_link_libraries(pr2_kinect_arm_track_human_hand ${catkin_LIBRARIES} ccd)

add_executable(arm_head_preparation src/arm_head_preparation.cpp)
target_link_libraries(arm_head_preparation ${catkin_LIBRARIES})
