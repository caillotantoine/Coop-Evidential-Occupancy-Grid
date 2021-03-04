cmake_minimum_required(VERSION 3.0.2)
project(perceptive_stream)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  image_transport
  cv_bridge
  geometry_msgs
  sensor_msgs
)

find_package(OpenCV 4 REQUIRED)


add_message_files(
  FILES
  Num.msg
  Img.msg
  BBox3D.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(cameraPublisher src/cameraPublisher.cpp)
target_link_libraries(cameraPublisher ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(cameraPublisher perceptive_stream_generate_messages_cpp)

catkin_install_python(PROGRAMS scripts/readCamStream.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/gndTruthCAR.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/bbox_on_img.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/utils.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)