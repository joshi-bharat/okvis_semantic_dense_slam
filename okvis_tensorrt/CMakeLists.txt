cmake_minimum_required(VERSION 3.8...3.30)

# TensorRT engine wrapper, shared by the SuperPoint frontend and the stereo depth network. It
# lives below both in the dependency graph: okvis_frontend cannot link okvis_deep_learning,
# which itself depends on the frontend.

find_package(CUDAToolkit REQUIRED)
find_package(Glog REQUIRED)
find_package(OpenCV REQUIRED core)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
    PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu)
# plain libnvinfer.so comes with libnvinfer-dev; fall back to the versioned runtime library
find_library(TENSORRT_LIBRARY
    NAMES nvinfer libnvinfer.so.10
    HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "USE_TENSORRT: TensorRT not found (NvInfer.h: ${TENSORRT_INCLUDE_DIR}, "
        "nvinfer: ${TENSORRT_LIBRARY}). Install libnvinfer-dev matching the installed runtime, "
        "or set TENSORRT_ROOT.")
endif()
message(STATUS "TensorRT: ${TENSORRT_INCLUDE_DIR}, ${TENSORRT_LIBRARY}")

set(LIB_NAME okvis_tensorrt)
add_library(${LIB_NAME}
  include/okvis/TensorRtEngine.hpp
  src/TensorRtEngine.cpp
)
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${GLOG_INCLUDE_DIRS}
)
target_include_directories(${LIB_NAME} SYSTEM PUBLIC $<BUILD_INTERFACE:${TENSORRT_INCLUDE_DIR}>)
target_link_libraries(${LIB_NAME}
  PUBLIC
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    CUDA::cudart
  PRIVATE
    ${GLOG_LIBRARIES}
)
target_compile_features(${LIB_NAME} PUBLIC cxx_std_${OKVIS_CXX_STANDARD})
target_compile_options(${LIB_NAME}
  PUBLIC
    ${OKVIS_PUBLIC_CXX_FLAGS}
  PRIVATE
    ${OKVIS_PRIVATE_CXX_FLAGS}
)
target_compile_definitions(${LIB_NAME} PUBLIC OKVIS_USE_TENSORRT)
add_library(okvis::TensorRt ALIAS ${LIB_NAME})

install(TARGETS ${LIB_NAME} EXPORT okvisTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/okvis DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")
