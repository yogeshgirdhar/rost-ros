cmake_minimum_required(VERSION 2.4.6)
option(BUILD_ROS_NODES "Build ROS nodes" ON)
option(BUILD_CLI_TOOLS "Build command line tools" ON)
option(USEGCC47  "Build using gcc4.7" OFF)


if(BUILD_ROS_NODES)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()
else()
  message("NOT building ROS nodes.")
endif()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)



#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

if(USEGCC47)
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -D_GLIBCXX_USE_NANOSLEEP")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -pthread")
endif()

if(BUILD_CLI_TOOLS)
  set(Boost_USE_STATIC_LIBS   ON)
  find_package(Boost COMPONENTS program_options)
  include_directories(${Boost_INCLUDE_DIR})
  include_directories(${Project_SOURCE_DIR}/include)
  
  add_executable(test_neighbors src/cli/test_neighbors.cpp )
  add_executable(test_rost src/cli/test_rost.cpp )
  add_executable(test_rost_online src/cli/test_rost_online.cpp)
  add_executable(rost_1d src/cli/rost_1d.cpp)
  add_executable(irost_1d src/cli/irost_1d.cpp)
  add_executable(generate_random_words src/cli/generate_random_words.cpp)  
  target_link_libraries(rost_1d ${Boost_PROGRAM_OPTIONS_LIBRARY} )
  target_link_libraries(irost_1d ${Boost_PROGRAM_OPTIONS_LIBRARY} )
endif()


if(BUILD_ROS_NODES)
  rosbuild_add_executable(rost_1d_node src/nodes/rost_1d_node.cpp)
  rosbuild_add_executable(rost_txy_image_node src/nodes/rost_txy_image_node.cpp)
endif()
