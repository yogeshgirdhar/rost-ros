cmake_minimum_required(VERSION 2.4.6)



include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)
#set(ROS_BUILD_TYPE Debug)
#set(ROS_BUILD_TYPE Release)
rosbuild_init()


include(ExternalProject)

set(EXT_PREFIX "${PROJECT_SOURCE_DIR}/ext")
set(FLANN_CMAKE_ARGS -DBUILD_MATLAB_BINDINGS=OFF -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_CUDA_LIB=OFF -DBUILD_C_BINDINGS=OFF -DCMAKE_INSTALL_PREFIX=${EXT_PREFIX})

ExternalProject_Add(FLANN_EXT
  PREFIX ${EXT_PREFIX}
  URL ${CMAKE_CURRENT_SOURCE_DIR}/ext/flann.tar.gz
  CMAKE_ARGS ${FLANN_CMAKE_ARGS}
)




#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
rosbuild_genmsg()
#uncomment if you have defined services
rosbuild_gensrv()


set(CMAKE_MODULE_PATH  "${PROJECT_SOURCE_DIR}/cmake")
find_package(OpenCV REQUIRED )
#find_package(Flann)

include_directories(${PROJECT_SOURCE_DIR})
include_directories("${PROJECT_SOURCE_DIR}/ext/include")
if(APPLE)
	set(FLANN_LIB "${PROJECT_SOURCE_DIR}/ext/lib/libflann_cpp.dylib")
else()
	set(FLANN_LIB "${PROJECT_SOURCE_DIR}/ext/lib/libflann_cpp.so")
endif()



#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()

#rosbuild_add_executable(image_source src/image_source.cpp)
rosbuild_add_executable(bow src/bow.cpp)
rosbuild_add_executable(bowtk src/bowtk.cpp src/image_source.cpp)
rosbuild_add_executable(orbtk src/orbtk.cpp src/image_source.cpp src/binary_features.cpp)
rosbuild_add_executable(image_cache src/image_cache.cpp)
rosbuild_link_boost(bow system)
rosbuild_link_boost(bowtk program_options)
rosbuild_link_boost(orbtk program_options)
#target_link_libraries(example ${PROJECT_NAME})
target_link_libraries(bow ${OpenCV_LIBS} ${FLANN_LIB})
target_link_libraries(bowtk ${OpenCV_LIBS})
target_link_libraries(orbtk ${OpenCV_LIBS})
