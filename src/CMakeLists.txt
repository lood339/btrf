cmake_minimum_required(VERSION 3.0)

set(SOURCES 

    yael_io.cpp
    btrf_util.cpp
    btrf_param.cpp
    btrf_tree_node.cpp
    btrf_tree.cpp
    btrf_forest.cpp
    btrf_forest_builder.cpp

    opencv_util/cvx_wht.cpp
    opencv_util/ms7scenes_util.cpp
    opencv_util/cvx_io.cpp
    opencv_util/cvx_util.cpp

    dt_common/dt_util.cpp
    dt_common/dt_random.cpp
    dt_common/vnl_random.cxx

    pose_estimation/cvx_pose_estimation.cpp
    pose_estimation/Kabsch.cpp
    pose_estimation/cvx_calib3d.cpp

    Walsh_Hadamard/wh.c
    Walsh_Hadamard/defs.c
    Walsh_Hadamard/image.c
    Walsh_Hadamard/matrix.c
)

set(HEADERS

    yael_io.hpp
    btrf_util.hpp
    btrf_param.hpp
    btrf_tree_node.hpp
    btrf_tree.hpp
    btrf_forest.hpp
    btrf_forest_builder.hpp

    opencv_util/dataset_param.hpp
    opencv_util/cvx_io.hpp
    opencv_util/cvx_util.hpp
    opencv_util/cvx_wht.hpp
    opencv_util/cvx_image_310.hpp
    opencv_util/ms7scenes_util.hpp

    dt_common/dt_random.hpp
    dt_common/vnl_random.h
    dt_common/dt_util.hpp

    pose_estimation/cvx_pose_estimation.hpp
    pose_estimation/cvx_calib3d.hpp
    pose_estimation/Kabsch.hpp

    Walsh_Hadamard/image.h
    Walsh_Hadamard/defs.h
    Walsh_Hadamard/matrix.h
    Walsh_Hadamard/wh.h
)

include_directories(../src)
include_directories(../src/opencv_util)
include_directories(../src/dt_common)
include_directories(../src/pose_estimation)
include_directories(../src/Walsh_Hadamard)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_library(btrf ${SOURCES} ${HEADERS})
target_link_libraries(btrf ${Eigen3_LIBRARIES} ${OpenCV_LIBRARIES})

set(CMD_UTILS
    btrf_train
    btrf_test
    btrf_pose_estimation
)

foreach(cmd_util ${CMD_UTILS})
    add_executable(${cmd_util} cmd/${cmd_util}.cpp)
    target_link_libraries(${cmd_util} btrf) 
endforeach()

