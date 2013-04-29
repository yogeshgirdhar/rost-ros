cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

include(ExternalProject)

set(EXT_PREFIX "${PROJECT_SOURCE_DIR}/ext")
set(YAMLCPP_CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXT_PREFIX})

ExternalProject_Add(YAMLCPP_EXT
  PREFIX ${EXT_PREFIX}
  URL ${CMAKE_CURRENT_SOURCE_DIR}/ext/yaml-cpp
  CMAKE_ARGS ${YAMLCPP_CMAKE_ARGS}
)
find_package(PkgConfig)
#pkg_search_module(TINYXML REQUIRED tinyxml)


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


FIND_PACKAGE( OpenCV REQUIRED )
rosbuild_add_boost_directories()
include_directories(${PROJECT_SOURCE_DIR})
include_directories("${EXT_PREFIX}/include")
link_directories(${EXT_PREFIX}/lib)


rosbuild_add_executable(visualwords src/visualwords.cpp src/video_writer.hpp)
rosbuild_add_executable(summary src/summary.cpp)

rosbuild_add_executable(genBrowser src/generateBrowser.cpp)
target_link_libraries(genBrowser yaml-cpp)
target_link_libraries(genBrowser ctemplate_nothreads)
rosbuild_link_boost(genBrowser system)
rosbuild_link_boost(genBrowser filesystem)

target_link_libraries(visualwords ${OpenCV_LIBS})
target_link_libraries(summary ${OpenCV_LIBS})
