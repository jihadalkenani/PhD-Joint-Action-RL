cmake_minimum_required(VERSION 3.0.2)
project(joint_action_rl)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(robot_execution_node src/cpp_node/robot_execution_node.cpp)
target_link_libraries(robot_execution_node ${catkin_LIBRARIES})
