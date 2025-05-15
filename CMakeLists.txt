cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)
project(detector_dl)
add_definitions(-w)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(CUDA REQUIRED)
ament_auto_find_build_dependencies()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DMY_DEF=1")
SET(CUDA_NVCC_FLAGS "-arch=sm_87" CACHE STRING "nvcc flags" FORCE)
SET(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)

cuda_add_library(detector_core SHARED
  src/cu/Detector.cpp
  src/cu/postprocess.cu
  src/cu/preprocess.cu
)

#cuda & tensorrt
target_include_directories(detector_core PUBLIC
  /usr/local/cuda-12.2/include
  /usr/include/aarch64-linux-gnu
  include
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(detector_core PUBLIC
  /usr/local/cuda-12.2/lib64
  /usr/lib/aarch64-linux-gnu
)

target_link_libraries(detector_core
  nvinfer
  cudart
  nvonnxparser
  ${OpenCV_LIBS}
)

ament_auto_add_library(${PROJECT_NAME} SHARED 
  src/DetectorNode.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  detector_core
  ${OpenCV_LIBS}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN DetectorDlNode
  EXECUTABLE ${PROJECT_NAME}_node
)

install(TARGETS
  detector_core
  DESTINATION lib
)

ament_auto_package(
  INSTALL_TO_SHARE
)


