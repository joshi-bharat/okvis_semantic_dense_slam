cmake_minimum_required(VERSION 3.8...3.30)

find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)
if(USE_SYSTEM_BRISK)
  find_package(brisk REQUIRED)
endif()
if(USE_SYSTEM_DBOW2)
  find_package(DBoW2 REQUIRED)
endif()
if(USE_SYSTEM_CERES)
  find_package(Ceres REQUIRED)
endif()
if(USE_NN AND NOT Torch_FOUND)
  find_package(Torch REQUIRED)
endif()

set(LIB_NAME okvis_frontend)
add_library(${LIB_NAME}
  include/DBoW2/FBrisk.hpp
  include/okvis/triangulation/stereo_triangulation.hpp
  include/opengv/absolute_pose/FrameNoncentralAbsoluteAdapter.hpp
  include/opengv/absolute_pose/LoopclosureNoncentralAbsoluteAdapter.hpp
  include/opengv/relative_pose/FrameRelativeAdapter.hpp
  include/opengv/sac_problems/absolute_pose/FrameAbsolutePoseSacProblem.hpp
  include/opengv/sac_problems/relative_pose/FrameRelativePoseSacProblem.hpp
  include/opengv/sac_problems/relative_pose/FrameRotationOnlySacProblem.hpp
  include/okvis/SuperPointDescriptors.hpp
  include/okvis/SuperPointNms.hpp
  include/okvis/SuperPointSparse.hpp
  src/FBrisk.cpp
  src/SuperPointDescriptors.cpp
  src/SuperPointNms.cpp
  src/SuperPointSparse.cpp
  src/FrameNoncentralAbsoluteAdapter.cpp
  src/FrameRelativeAdapter.cpp
  src/Frontend.cpp
  src/LoopclosureNoncentralAbsoluteAdapter.cpp
  src/stereo_triangulation.cpp
)
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${DBoW2_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${OPENGV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(${LIB_NAME}
  PUBLIC
    Eigen3::Eigen
    okvis::Ceres
    okvis::CV
    okvis::Kinematics
    okvis::Timing
    okvis::Util
    brisk::brisk
    ${DBoW2_LIBRARIES}
  PRIVATE
    ${GLOG_LIBRARIES}
    ${OPENGV_LIBRARIES}
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    Ceres::ceres
)
target_compile_features(${LIB_NAME} PUBLIC cxx_std_${OKVIS_CXX_STANDARD})
target_compile_options(${LIB_NAME}
  PUBLIC
    ${OKVIS_PUBLIC_CXX_FLAGS}
  PRIVATE
    ${OKVIS_PRIVATE_CXX_FLAGS}
)
if(USE_NN)
  target_compile_definitions(${LIB_NAME} PUBLIC OKVIS_USE_NN C10_USE_GLOG)
  target_compile_options(${LIB_NAME} PUBLIC ${TORCH_CXX_FLAGS})
  if(USE_GPU)
    target_compile_definitions(${LIB_NAME} PUBLIC OKVIS_USE_GPU)
  endif()
endif()
# SuperPoint runs on the GPU, so its matcher and extractor are only built with TensorRT. Without
# them the frontend is BRISK-only.
if(USE_TENSORRT)
  find_package(CUDAToolkit REQUIRED)
  target_sources(${LIB_NAME} PRIVATE
    include/okvis/CublasMatcher.hpp
    src/CublasMatcher.cpp
    include/okvis/SuperPointExtractor.hpp
    src/SuperPointExtractor.cpp
  )
  target_link_libraries(${LIB_NAME} PUBLIC okvis::TensorRt PRIVATE CUDA::cublas)
  target_compile_definitions(${LIB_NAME} PUBLIC OKVIS_USE_SUPERPOINT)
endif()
add_library(okvis::Frontend ALIAS ${LIB_NAME})

# the DBoW2 pre-built vocabulary
add_custom_command(TARGET ${LIB_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/../resources/small_voc.yml.gz
          ${CMAKE_CURRENT_BINARY_DIR}/..)

# and the network
add_custom_command(TARGET ${LIB_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/../resources/fast-scnn.pt
          ${CMAKE_CURRENT_BINARY_DIR}/..)

install(TARGETS ${LIB_NAME} EXPORT okvisTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/okvis DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")

# testing
if(BUILD_TESTS)
  if(APPLE)
    add_definitions(-DGTEST_HAS_TR1_TUPLE=1)
  else()
    add_definitions(-DGTEST_HAS_TR1_TUPLE=0)
  endif(APPLE)
  enable_testing()
  set(PROJECT_TEST_NAME ${LIB_NAME}_test)
  add_executable(${PROJECT_TEST_NAME}
    test/runTests.cpp
    test/TestSuperPointParameters.cpp
    test/TestSuperPointNms.cpp
    test/TestSuperPointDescriptors.cpp
  )
  if(USE_TENSORRT)
    target_sources(${PROJECT_TEST_NAME} PRIVATE test/TestCublasMatcher.cpp test/TestSuperPointExtractor.cpp
      test/TestSuperPointSparse.cpp)
  endif()
  target_link_libraries(${PROJECT_TEST_NAME}
    ${LIB_NAME}
    okvis::Common
    gtest
  )
  target_compile_options(${PROJECT_TEST_NAME}
    PUBLIC
    ${OKVIS_PUBLIC_CXX_FLAGS}
    PRIVATE
    ${OKVIS_PRIVATE_CXX_FLAGS}
  )
  # The parameter reader needs a complete config, so the tests build their fixtures from the
  # real ones rather than from hand-written stubs.
  target_compile_definitions(${PROJECT_TEST_NAME}
    PRIVATE OKVIS_TEST_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../config"
  )
  add_test(NAME ${PROJECT_TEST_NAME} COMMAND ${PROJECT_TEST_NAME})
endif()
