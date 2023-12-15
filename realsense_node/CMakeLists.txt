cmake_minimum_required(VERSION 3.0.2)
project(realsense_node)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)





include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
