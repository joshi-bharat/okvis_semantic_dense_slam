cmake_minimum_required(VERSION 3.8...3.30)
project(okvis VERSION 2.0.0 LANGUAGES CXX C)

# options for build configuration
option (USE_SYSTEM_BRISK
        "Use brisk via find_package rather than downloading it as part of okvis" OFF)
option (USE_SYSTEM_DBOW2
        "Use DBoW2 via find_package rather than downloading it as part of okvis" OFF)
option (USE_SYSTEM_CERES
        "Use ceres via find_package rather than downloading it as part of okvis" OFF)
option (BUILD_APPS
        "Builds a demo app (which require boost and OpenCV highgui)" ON)
option (BUILD_TESTS
        "Builds all gtests" OFF)
option (BUILD_ROS2
        "Builds ros2 wrapper" ON)
option (HAVE_LIBREALSENSE
        "Use realsense as part of okvis" OFF)
option (USE_NN
        "Use keypoint classification as part of okvis, requires torch" ON)
option (USE_GPU
        "Use keypoint classification with GPU inference, requires torch with Cuda" ON)
option (DO_TIMING
        "Enable timing" ON)
option (BUILD_SHARED_LIBS
        "Whether to build okvis as a shared library" OFF)

option (USE_COLIDMAP "Whether to build okvis supportin colour and object's id 'for language based OKVIS' or regular one" ON)
option (USE_RERUN "Log the estimate to a Rerun viewer (downloads the Rerun C++ SDK and Arrow)" ON)
option (USE_TENSORRT "Stereo depth network via a TensorRT engine (depth-model.engine), without libtorch" ON)
# The viewer (pip install rerun-sdk==<version>) must match the SDK version.
# 0.38.1 has no GitHub release asset yet; this is Rerun's C++ SDK build of the 0.38.1 release commit
# (b08c599, the same commit as `rerun --version` of pip rerun-sdk 0.38.1). For a GitHub release use
# https://github.com/rerun-io/rerun/releases/download/<version>/rerun_cpp_sdk.zip
set(RERUN_SDK_URL "https://build.rerun.io/commit/b08c599/rerun_cpp_sdk.zip"
    CACHE STRING "Rerun C++ SDK zip (0.38.1)")

# C++20 throughout: LibTorch >= 2.14 refuses to compile as C++17. Set globally as well as on the
# okvis targets so the vendored subprojects (ceres, supereight2, brisk, DBoW2, ...) match and the
# ABI stays consistent across the whole build.
set(OKVIS_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD ${OKVIS_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# LibTorch lives outside the default loader paths, so bake the directories of everything we link
# against into the installed binaries' RPATH. Without this they fail at startup with
# "libtorch.so: cannot open shared object file".
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(OKVIS_PUBLIC_CXX_FLAGS)
if(APPLE)
  set(OKVIS_PRIVATE_CXX_FLAGS -Wall -Wpedantic -Wextra -Wno-dtor-name)
else()
  set(OKVIS_PRIVATE_CXX_FLAGS -Wall -Wpedantic -Wextra)
endif()

if(NOT DEFINED BLA_VENDOR)
  set(BLA_VENDOR OpenBLAS)
endif()
find_package(LAPACK REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "/usr/share/cmake/geographiclib")

# Make sure we use Release and warn otherwise
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  message(WARNING "CMAKE_BUILD_TYPE not set to 'Release'. Performance may be terrible.")
endif()

if(USE_NN AND USE_GPU)
  # Build for the GPU in this machine unless asked otherwise. Set -DCMAKE_CUDA_ARCHITECTURES=<sm>
  # explicitly when cross-building or shipping binaries for other cards.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
endif()

if(USE_NN AND USE_GPU)
  if(APPLE)
    # MPS enabled by default
    add_compile_definitions(OKVIS_USE_MPS)
  else()
    # If CUDA isn't enabled explicitly here it's not detected correctly by Torch.
    enable_language(CUDA)
    add_compile_definitions(OKVIS_USE_CUDA)
  endif()
endif()

if(USE_NN)
  # LibTorch is usually not in CMAKE_PREFIX_PATH; fall back to ~/libtorch, which is where the
  # README tells you to unpack it. Set -DTorch_DIR=... or the Torch_DIR env var to override.
  if(NOT Torch_DIR AND NOT DEFINED ENV{Torch_DIR}
     AND EXISTS "$ENV{HOME}/libtorch/share/cmake/Torch/TorchConfig.cmake")
    set(Torch_DIR "$ENV{HOME}/libtorch/share/cmake/Torch")
    message(STATUS "Torch_DIR not given, using ${Torch_DIR}")
  endif()
  find_package(Torch REQUIRED)

  # Function to check if file exists and has correct hash
  function(check_hash file expected_hash result_var)
      if(EXISTS "${file}")
          file(SHA256 "${file}" actual_hash)
          if("${expected_hash}" STREQUAL "SHA256=${actual_hash}")
              set(${result_var} TRUE PARENT_SCOPE)
              return()
          endif()
      endif()
      set(${result_var} FALSE PARENT_SCOPE)
  endfunction()

  # Download the models.
  message(STATUS "Downloading depth models")

  # Stereo depth prediction (unimatch). Two are published; DEPTH_MODEL picks which one ends up as
  # depth-model.pt, the name the stereo processor loads. Only the selected one is downloaded.
  #   fast     "stereo-indoor-sigma.pt", used for the euroc and hilti22 datasets
  #   accuracy "stereo-mix-sigma.pt", used for the VBR dataset
  set(DEPTH_MODEL "fast" CACHE STRING "Stereo depth model to use: fast | accuracy")
  set_property(CACHE DEPTH_MODEL PROPERTY STRINGS fast accuracy)

  if(DEPTH_MODEL STREQUAL "fast")
    set(DEPTH_URL "https://cvg.cit.tum.de/webshare/g/srl-models/unimatch/stereo-indoor-sigma.pt")
    set(DEPTH_SOURCE_FILE "${PROJECT_SOURCE_DIR}/build/stereo-indoor-sigma.pt")
    set(DEPTH_HASH "SHA256=75d3a68b0a18e8ff1b20ac1e4b75a094d595c57cf52482485d4e8049782fe4c8")
  elseif(DEPTH_MODEL STREQUAL "accuracy")
    set(DEPTH_URL "https://cvg.cit.tum.de/webshare/g/srl-models/unimatch/stereo-mix-sigma.pt")
    set(DEPTH_SOURCE_FILE "${PROJECT_SOURCE_DIR}/build/stereo-mix-sigma.pt")
    set(DEPTH_HASH "SHA256=33e7016865560adca17e4402624abfa495d0016ea0a6fb3091c18ed73f439165")
  else()
    message(FATAL_ERROR "DEPTH_MODEL must be 'fast' or 'accuracy', got '${DEPTH_MODEL}'")
  endif()
  set(DEPTH_FILE "${PROJECT_SOURCE_DIR}/build/depth-model.pt")

  # Nothing to do if depth-model.pt is already the selected model. Checking the destination and
  # not just the download keeps reconfiguring free, and catches a truncated or mismatched file.
  check_hash("${DEPTH_FILE}" "${DEPTH_HASH}" DEPTH_INSTALLED_OK)
  if(DEPTH_INSTALLED_OK)
    message(STATUS "Depth model (${DEPTH_MODEL}) already in place. No download needed!")
  else()
    check_hash("${DEPTH_SOURCE_FILE}" "${DEPTH_HASH}" DEPTH_SOURCE_OK)
    if(NOT DEPTH_SOURCE_OK)
      message(STATUS "Downloading depth model (${DEPTH_MODEL})...")
      file(DOWNLOAD
        ${DEPTH_URL}
        ${DEPTH_SOURCE_FILE}
        EXPECTED_HASH ${DEPTH_HASH}
        TIMEOUT 600
        STATUS DEPTH_DOWNLOAD_STATUS
      )
      list(GET DEPTH_DOWNLOAD_STATUS 0 DEPTH_DOWNLOAD_CODE)
      if(NOT DEPTH_DOWNLOAD_CODE EQUAL 0)
        list(GET DEPTH_DOWNLOAD_STATUS 1 DEPTH_DOWNLOAD_MESSAGE)
        # Remove the partial file: leaving it behind makes the next configure believe it has a
        # download to work with, and it would be copied over a good depth-model.pt.
        file(REMOVE "${DEPTH_SOURCE_FILE}")
        message(FATAL_ERROR
          "Failed to download ${DEPTH_URL}: ${DEPTH_DOWNLOAD_MESSAGE}\n"
          "Fetch it by hand into ${DEPTH_SOURCE_FILE}, or configure with -DUSE_NN=OFF.")
      endif()
    endif()
    # Copy rather than rename, so the source file survives for the next configure.
    file(COPY_FILE "${DEPTH_SOURCE_FILE}" "${DEPTH_FILE}")
  endif()

  if(BUILD_ROS2)
    file(COPY ${DEPTH_FILE}
        DESTINATION ${PROJECT_SOURCE_DIR}/resources/)
  endif()

  # Get Model for Depth Prediction: Multi View Stereo
  set(MVS_URL "https://cvg.cit.tum.de/webshare/g/srl-models/mvs/mvs-sigma.pt")
  set(MVS_FILE "${PROJECT_SOURCE_DIR}/build/mvs-model.pt")
  set(MVS_HASH "SHA256=1f37ee7499b1b38dfdb11208b859cf02c127e5f120e9db3ed34847ff14eb6633")
  check_hash("${MVS_FILE}" "${MVS_HASH}" MVS_OK)
  if(MVS_OK)
    message(STATUS "MVS Model already available. No Download needed!")
  else()
    file(DOWNLOAD
      ${MVS_URL}
      ${MVS_FILE}
      EXPECTED_HASH ${MVS_HASH}
      TIMEOUT 120
    )
  endif()
  if(BUILD_ROS2)
    file(COPY ${MVS_FILE}
        DESTINATION ${PROJECT_SOURCE_DIR}/resources/)
  endif()

  if(USE_COLIDMAP)
    # Models for FindAnything: Efficient SAM. Two are published; ESAM_MODEL picks which one ends
    # up as esam-model.pt, the name VLProcessor loads. Only the selected one is downloaded.
    set(ESAM_MODEL "original" CACHE STRING "Efficient SAM model to use: original | fast")
    set_property(CACHE ESAM_MODEL PROPERTY STRINGS original fast)

    if(ESAM_MODEL STREQUAL "fast")
      set(ESAM_URL "https://cvg.cit.tum.de/webshare/g/srl-models/findanything/esam-model-fast.pt")
      set(ESAM_SOURCE_FILE "${PROJECT_SOURCE_DIR}/build/esam-model-fast.pt")
      set(ESAM_HASH "SHA256=dd76bfa171b70ea580be5361c04f936051f1be5780e57e369c310c8610c24d0f")
    elseif(ESAM_MODEL STREQUAL "original")
      set(ESAM_URL "https://cvg.cit.tum.de/webshare/g/srl-models/findanything/esam-model-original.pt")
      set(ESAM_SOURCE_FILE "${PROJECT_SOURCE_DIR}/build/esam-model-original.pt")
      set(ESAM_HASH "SHA256=428d6c0409fffa6f9b5ce713ca635d22b2f4800817c1e4daed323d5c2e38735e")
    else()
      message(FATAL_ERROR "ESAM_MODEL must be 'original' or 'fast', got '${ESAM_MODEL}'")
    endif()
    set(ESAM_FILE "${PROJECT_SOURCE_DIR}/build/esam-model.pt")

    # Nothing to do if esam-model.pt is already the selected model. Checking the destination and
    # not just the download keeps reconfiguring free, and catches a truncated or mismatched file.
    check_hash("${ESAM_FILE}" "${ESAM_HASH}" ESAM_INSTALLED_OK)
    if(ESAM_INSTALLED_OK)
      message(STATUS "ESAM model (${ESAM_MODEL}) already in place. No download needed!")
    else()
      check_hash("${ESAM_SOURCE_FILE}" "${ESAM_HASH}" ESAM_SOURCE_OK)
      if(NOT ESAM_SOURCE_OK)
        message(STATUS "Downloading ESAM model (${ESAM_MODEL}), this may take a while...")
        file(DOWNLOAD
          ${ESAM_URL}
          ${ESAM_SOURCE_FILE}
          EXPECTED_HASH ${ESAM_HASH}
          TIMEOUT 600
          STATUS ESAM_DOWNLOAD_STATUS
        )
        list(GET ESAM_DOWNLOAD_STATUS 0 ESAM_DOWNLOAD_CODE)
        if(NOT ESAM_DOWNLOAD_CODE EQUAL 0)
          list(GET ESAM_DOWNLOAD_STATUS 1 ESAM_DOWNLOAD_MESSAGE)
          # Remove the partial file: leaving it behind makes the next configure believe it has a
          # download to work with, and it would be copied over a good esam-model.pt.
          file(REMOVE "${ESAM_SOURCE_FILE}")
          message(FATAL_ERROR
            "Failed to download ${ESAM_URL}: ${ESAM_DOWNLOAD_MESSAGE}\n"
            "Fetch it by hand into ${ESAM_SOURCE_FILE}, or configure with -DUSE_COLIDMAP=OFF "
            "to build without FindAnything.")
        endif()
      endif()
      # Copy rather than rename, so the source file survives for the next configure.
      file(COPY_FILE "${ESAM_SOURCE_FILE}" "${ESAM_FILE}")
    endif()

    if(BUILD_ROS2)
      file(COPY ${ESAM_FILE}
          DESTINATION ${PROJECT_SOURCE_DIR}/resources/)
    endif()

    # Get Models for FindAnything: CLIP Vision
    set(CLIP_URL "https://cvg.cit.tum.de/webshare/g/srl-models/findanything/vl-vision-model.pt")
    set(CLIP_FILE "${PROJECT_SOURCE_DIR}/build/vl-vision-model.pt")
    set(CLIP_HASH "SHA256=fbd6423f00f9389a3c9ada9d5b49205ac935afcc6a32230d2c061f057e3907fa")
    check_hash("${CLIP_FILE}" "${CLIP_HASH}" CLIP_OK)
    if(CLIP_OK)
      message(STATUS "CLIP Vision Model already available. No Download needed!")
    else()
      file(DOWNLOAD
        ${CLIP_URL}
        ${CLIP_FILE}
        EXPECTED_HASH ${CLIP_HASH}
        TIMEOUT 500
      )
    endif()
    if(BUILD_ROS2)
      file(COPY ${CLIP_FILE}
          DESTINATION ${PROJECT_SOURCE_DIR}/resources/)
    endif()
  endif()
endif()

if(NOT DO_TIMING)
  add_compile_definitions(DEACTIVATE_TIMERS)
  message(STATUS "Deactivating timers")
endif()

if(BUILD_TESTS)
  # Must be called before adding any tests for them to be registered with CMake
  enable_testing()
endif()

add_subdirectory(external)
# SE2 Stuff
option(SE_TEST "Compile the supereight tests" OFF)
option(SE_APP "Compile the supereight application" OFF)
add_subdirectory(supereight2)


add_subdirectory(okvis_util)
add_subdirectory(okvis_kinematics)
add_subdirectory(okvis_time)
add_subdirectory(okvis_cv)
add_subdirectory(okvis_common)
add_subdirectory(okvis_timing)
add_subdirectory(okvis_ceres)
add_subdirectory(okvis_frontend)
add_subdirectory(okvis_mapping)
if(USE_RERUN)
  include(FetchContent)
  FetchContent_Declare(rerun_sdk
    URL ${RERUN_SDK_URL})
  FetchContent_MakeAvailable(rerun_sdk)
  message(STATUS "Rerun visualisation enabled (SDK ${RERUN_SDK_URL})")
endif()
add_subdirectory(okvis_multisensor_processing)

# Include stereo from depth network.
if(USE_NN OR USE_TENSORRT)
  add_subdirectory(okvis_deep_learning)
endif()

if(BUILD_APPS)
  find_package(Boost REQUIRED COMPONENTS filesystem)
  find_package(OpenCV REQUIRED core imgcodecs highgui calib3d)
  if(USE_SYSTEM_BRISK)
    find_package(brisk REQUIRED)
  endif()
  if(USE_SYSTEM_DBOW2)
    find_package(DBoW2 REQUIRED)
  endif()
  
  # DBoW2 test app
  add_executable(dbow2_test okvis_apps/src/dbow2_test.cpp)
  target_include_directories(dbow2_test
    PRIVATE
      ${DBoW2_INCLUDE_DIRS}
  )
  target_link_libraries(dbow2_test PRIVATE
    ${OpenCV_LIBS}
    Boost::filesystem
    okvis::Frontend
  )
  target_compile_options(dbow2_test
    PUBLIC
      ${OKVIS_PUBLIC_CXX_FLAGS}
    PRIVATE
      ${OKVIS_PRIVATE_CXX_FLAGS}
  )


  # require GeographicLib
  find_package(GeographicLib REQUIRED)

  # OKVIS2 app
  add_executable(okvis_app_synchronous okvis_apps/src/okvis_app_synchronous.cpp)
  target_link_libraries(okvis_app_synchronous
    ${OpenCV_LIBS}
    Boost::filesystem
    okvis::MultisensorProcessing
  )
  target_compile_options(okvis_app_synchronous
    PUBLIC
      ${OKVIS_PUBLIC_CXX_FLAGS}
    PRIVATE
      ${OKVIS_PRIVATE_CXX_FLAGS}
  )
  install(TARGETS okvis_app_synchronous EXPORT okvisTargets RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_executable(okvis2x_app_synchronous
          okvis_apps/src/okvis2x_app_synchronous.cpp
          )
  target_link_libraries(okvis2x_app_synchronous
          okvis::Util
          okvis::Kinematics
          okvis::Time
          okvis::CV
          okvis::Common
          okvis::Ceres
          okvis::Timing
          okvis::Frontend
          okvis::MultisensorProcessing
          pthread
          ${Boost_LIBRARIES}
          ${TORCH_LIBRARIES}
          ${GeographicLib_LIBRARIES}
          SRL::Supereight2
          SRL::Projection
  )
  target_compile_options(okvis2x_app_synchronous
    PUBLIC
      ${OKVIS_PUBLIC_CXX_FLAGS}
    PRIVATE
      ${OKVIS_PRIVATE_CXX_FLAGS}
  )

  target_compile_features(okvis2x_app_synchronous
          PUBLIC cxx_std_${OKVIS_CXX_STANDARD}
  )

  if(HAVE_LIBREALSENSE)
    # OKIVS RealSense app
    add_executable(okvis_app_realsense okvis_apps/src/okvis_app_realsense.cpp)
    target_link_libraries(okvis_app_realsense
      ${OpenCV_LIBS}
      okvis::MultisensorProcessing
    )
    target_compile_options(okvis_app_realsense
      PUBLIC
        ${OKVIS_PUBLIC_CXX_FLAGS}
      PRIVATE
        ${OKVIS_PRIVATE_CXX_FLAGS}
    )
    install(TARGETS okvis_app_realsense EXPORT okvisTargets RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  
    # OKIVS RealSense recorder app
    add_executable(okvis_app_realsense_recorder okvis_apps/src/okvis_app_realsense_recorder.cpp)
    target_link_libraries(okvis_app_realsense_recorder
      ${OpenCV_LIBS}
      okvis::MultisensorProcessing
    )
    target_compile_options(okvis_app_realsense_recorder
      PUBLIC
        ${OKVIS_PUBLIC_CXX_FLAGS}
      PRIVATE
        ${OKVIS_PRIVATE_CXX_FLAGS}
    )
    install(TARGETS okvis_app_realsense_recorder EXPORT okvisTargets RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
  
  if(USE_NN)
    find_package(Torch REQUIRED)
  
    # Torch test app
    add_executable(nn_test okvis_apps/src/nn_test.cpp)
    target_compile_options(nn_test PRIVATE ${TORCH_CXX_FLAGS})
    target_include_directories(nn_test
      PRIVATE
        ${TORCH_INCLUDE_DIRS}
    )
    target_link_libraries(nn_test
      ${OpenCV_LIBS}
      ${TORCH_LIBRARIES}
    )
    target_compile_features(nn_test PUBLIC cxx_std_${OKVIS_CXX_STANDARD})
    target_compile_options(nn_test
      PUBLIC
        ${OKVIS_PUBLIC_CXX_FLAGS}
      PRIVATE
        ${OKVIS_PRIVATE_CXX_FLAGS}
    )
    target_compile_features(nn_test
      PUBLIC cxx_std_${OKVIS_CXX_STANDARD}
    )
    
    add_executable(okvis2x_app_depthfusion_synchronous
                   okvis_apps/src/okvis2x_app_network_synchronous.cpp)
    target_compile_definitions(okvis2x_app_depthfusion_synchronous PUBLIC OKVIS_DFUSION_NETWORK_PROCESSOR)
    target_link_libraries(okvis2x_app_depthfusion_synchronous
      okvis_util
      okvis_kinematics
      okvis_time
      okvis_cv
      okvis_common
      okvis_ceres
      okvis_timing
      okvis_frontend
      okvis_multisensor_processing
      okvis_deep_learning
      pthread
      ${Boost_LIBRARIES}
      ${TORCH_LIBRARIES}
      ${GeographicLib_LIBRARIES}
      SRL::Supereight2
      SRL::Projection
    )

    add_executable(okvis2x_app_language_synchronous
                   okvis_apps/src/okvis2x_app_network_synchronous.cpp)
    target_compile_definitions(okvis2x_app_language_synchronous PUBLIC OKVIS_LANGUAGE_NETWORK_PROCESSOR)
    target_link_libraries(okvis2x_app_language_synchronous
      okvis_util
      okvis_kinematics
      okvis_time
      okvis_cv
      okvis_common
      okvis_ceres
      okvis_timing
      okvis_frontend
      okvis_multisensor_processing
      okvis_deep_learning
      okvis_mapping
      pthread
      ${Boost_LIBRARIES}
      ${TORCH_LIBRARIES}
      ${GeographicLib_LIBRARIES}
      SRL::Supereight2
      SRL::Projection
    )
  endif()               

  if(USE_NN OR USE_TENSORRT)
    # Submapping App with Stereo Depth NN
    add_executable(okvis2x_app_snetwork_synchronous
                   okvis_apps/src/okvis2x_app_network_synchronous.cpp)
    target_compile_definitions(okvis2x_app_snetwork_synchronous PUBLIC OKVIS_STEREO_NETWORK_PROCESSOR)
    target_link_libraries(okvis2x_app_snetwork_synchronous
      okvis_util
      okvis_kinematics
      okvis_time
      okvis_cv
      okvis_common
      okvis_ceres
      okvis_timing
      okvis_frontend
      okvis_multisensor_processing
      okvis_deep_learning
      okvis_mapping
      pthread
      ${Boost_LIBRARIES}
      ${TORCH_LIBRARIES}
      ${GeographicLib_LIBRARIES}
      SRL::Supereight2
      SRL::Projection
    )
  endif()
endif()

# ros2
if(BUILD_ROS2)
  if (NOT "$ENV{ROS_VERSION}" STREQUAL "2")
    message(WARNING "ROS2 not found, will build without")
    set(BUILD_ROS2 OFF)
  endif()
endif()
if(BUILD_ROS2)
  include_directories(okvis_ros2/include/)
  message(STATUS "Building ROS2 wrapper")

  # list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
  # ROS2 dependencies
  find_package(ament_cmake REQUIRED)
  find_package(ament_cmake_python REQUIRED)
  find_package(builtin_interfaces REQUIRED)
  find_package(eigen3_cmake_module REQUIRED)
  find_package(Eigen3 REQUIRED)
  find_package(OpenCV REQUIRED)
  find_package(cv_bridge REQUIRED)
  find_package(std_srvs REQUIRED)
  find_package(std_msgs REQUIRED)
  # only the vision-language (FindAnything) nodes need these messages
  if(USE_NN AND USE_COLIDMAP)
    find_package(language_feature_msgs REQUIRED)
  endif()
  find_package(sensor_msgs REQUIRED)
  find_package(visualization_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(Boost REQUIRED)
  find_package(tf2_eigen REQUIRED)
  find_package(tf2_geometry_msgs REQUIRED)

  find_package(PCL REQUIRED QUIET COMPONENTS common)
  include_directories(${PCL_INCLUDE_DIRS} include)
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})

  find_package(pcl_msgs REQUIRED)
  find_package(pcl_conversions REQUIRED)
  find_package(rclpy REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_components REQUIRED)
  find_package(rosbag2_cpp REQUIRED)
  find_package(rosbag2_storage REQUIRED)
  find_package(tf2 REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(image_transport REQUIRED)
  find_package(message_filters REQUIRED)

  # incldues and lib directories
  include_directories(${OpenCV_INCLUDE_DIRS})
  include_directories(${Eigen3_INCLUDE_DIRS})
  include_directories( include ${PCL_INCLUDE_DIRS} )

  # okvis::Ros2 library
  set(LIB_NAME okvis_ros2)
  add_library(${LIB_NAME}
    okvis_ros2/src/Subscriber.cpp
    okvis_ros2/src/RosbagReader.cpp
    okvis_ros2/src/Publisher.cpp
    okvis_ros2/src/RePublisher.cpp
  )
  target_include_directories(${LIB_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/okvis_ros2/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${GLOG_INCLUDE_DIRS}
  )
  if(USE_NN OR USE_TENSORRT)
    target_link_libraries(${LIB_NAME}
      okvis::DeepLearning
    )
  endif()

  target_link_libraries(${LIB_NAME}
      ${OpenCV_LIBS}
      Ceres::ceres
      Eigen3::Eigen
      SRL::Supereight2
      okvis::Common
      okvis::Util
      okvis::Timing
      okvis::Mapping
      okvis::MultisensorProcessing
      ${GLOG_LIBRARIES}
      ${Boost_LIBRARIES} 
      ${PCL_LIBRARIES}
  )
  target_compile_features(${LIB_NAME} PUBLIC cxx_std_${OKVIS_CXX_STANDARD})
  target_compile_options(${LIB_NAME}
    PUBLIC
      ${OKVIS_PUBLIC_CXX_FLAGS}
    PRIVATE
      ${OKVIS_PRIVATE_CXX_FLAGS}
  )
  if(OPENMP_FOUND)
    target_compile_definitions(${LIB_NAME} PRIVATE USE_OPENMP)
  endif()
  add_library(okvis::Ros2 ALIAS ${LIB_NAME})

  # target dependencies
  ament_target_dependencies(
    ${LIB_NAME}
    PCL
    ament_cmake
    rclcpp
    rclcpp_components
    rosbag2_cpp
    rosbag2_storage
    eigen3_cmake_module
    Eigen3
    sensor_msgs
    visualization_msgs
    std_srvs
    nav_msgs
    pcl_conversions
    message_filters
    cv_bridge
    image_transport
    tf2
    tf2_ros
    tf2_eigen
    tf2_geometry_msgs
  )

  # export dependencies
  ament_export_dependencies(
    PCL
    ament_cmake
    rclcpp
    rclcpp_components
    rosbag2_cpp
    rosbag2_storage
    eigen3_cmake_module
    Eigen3
    std_msgs
    std_srvs
    sensor_msgs
    visualization_msgs
    nav_msgs
    pcl_conversions
    message_filters
    cv_bridge
    image_transport
    tf2
    tf2_ros
  )

  install(TARGETS ${LIB_NAME} EXPORT okvisTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  install(DIRECTORY include/okvis DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")

  # OKVIS2X nodes - Wo/ Deep Learning

  # Subscriber
  add_executable(okvis2x_node_subscriber okvis_ros2/src/okvis2x_node.cpp)
  target_link_libraries(
    okvis2x_node_subscriber
    okvis::Ros2
    ${OpenCV_LIBS}
    ${SUITESPARSE_LIBRARIES}
    ${Boost_LIBRARIES} 
    ${PCL_LIBRARIES}
  )
  install(TARGETS okvis2x_node_subscriber DESTINATION lib/${PROJECT_NAME})

  # Synchronous (blocking) processing of a dataset folder or ROS2 bag read from disk
  add_executable(okvis_node_synchronous okvis_ros2/src/okvis_node_synchronous.cpp)
  target_link_libraries(
    okvis_node_synchronous
    okvis::Ros2
    ${OpenCV_LIBS}
    ${SUITESPARSE_LIBRARIES}
    ${Boost_LIBRARIES}
    ${PCL_LIBRARIES}
  )
  install(TARGETS okvis_node_synchronous DESTINATION lib/${PROJECT_NAME})
  # Realsense
  if(HAVE_LIBREALSENSE)

    add_executable(okvis_node_realsense_publisher okvis_ros2/src/okvis_node_realsense_publisher.cpp)
    target_link_libraries(
      okvis_node_realsense_publisher
      okvis::Ros2
      ${OpenCV_LIBS}
      ${SUITESPARSE_LIBRARIES}
      ${Boost_LIBRARIES} 
      ${PCL_LIBRARIES}
    )
    install(TARGETS okvis_node_realsense_publisher DESTINATION lib/${PROJECT_NAME})
    add_executable(okvis2x_node_realsense okvis_ros2/src/okvis2x_node.cpp)
    target_compile_definitions(okvis2x_node_realsense PUBLIC SRL_NAV_USE_REALSENSE)
    target_link_libraries(
      okvis2x_node_realsense
      okvis::Ros2
      ${OpenCV_LIBS}
      ${SUITESPARSE_LIBRARIES}
      ${Boost_LIBRARIES} 
      ${PCL_LIBRARIES}
    )
    install(TARGETS okvis2x_node_realsense DESTINATION lib/${PROJECT_NAME})
  endif()

  # OKVIS2X nodes - W/ Stereo Network (TorchScript or TensorRT backend)
  if(USE_NN OR USE_TENSORRT)
    # W/ Stereo Network Processor
    add_executable(okvis2x_stereo_network_node_subscriber okvis_ros2/src/okvis2x_network_node.cpp)
    target_compile_definitions(okvis2x_stereo_network_node_subscriber PUBLIC OKVIS_STEREO_NETWORK_PROCESSOR)
    target_link_libraries(
      okvis2x_stereo_network_node_subscriber
      okvis::Ros2
      ${OpenCV_LIBS}
      ${SUITESPARSE_LIBRARIES}
      ${Boost_LIBRARIES} 
      ${PCL_LIBRARIES}
      okvis::DeepLearning
    )
    install(TARGETS okvis2x_stereo_network_node_subscriber DESTINATION lib/${PROJECT_NAME})

    # W/ Stereo Network Processor, reading a ROS2 bag sequentially (no `ros2 bag play`)
    add_executable(okvis2x_stereo_network_node_bag okvis_ros2/src/okvis2x_network_node_bag.cpp)
    target_compile_definitions(okvis2x_stereo_network_node_bag PUBLIC OKVIS_STEREO_NETWORK_PROCESSOR)
    target_link_libraries(
      okvis2x_stereo_network_node_bag
      okvis::Ros2
      ${OpenCV_LIBS}
      ${SUITESPARSE_LIBRARIES}
      ${Boost_LIBRARIES}
      ${PCL_LIBRARIES}
      okvis::DeepLearning
    )
    install(TARGETS okvis2x_stereo_network_node_bag DESTINATION lib/${PROJECT_NAME})

    # W/ Stereo Network Processor AND Vision-Language Mapping (FindAnything), reading a ROS2 bag.
    # The stereo network provides the depth FindAnything needs, so no depth camera is required.
    if(USE_NN AND USE_COLIDMAP)
      add_executable(okvis2x_stereo_language_node_bag okvis_ros2/src/okvis2x_network_node_bag.cpp)
      target_compile_definitions(okvis2x_stereo_language_node_bag
                                 PUBLIC OKVIS_STEREO_NETWORK_PROCESSOR OKVIS_LANGUAGE_NETWORK_PROCESSOR)
      target_link_libraries(
        okvis2x_stereo_language_node_bag
        okvis::Ros2
        ${OpenCV_LIBS}
        ${SUITESPARSE_LIBRARIES}
        ${Boost_LIBRARIES}
        ${PCL_LIBRARIES}
        okvis::DeepLearning
      )
      install(TARGETS okvis2x_stereo_language_node_bag DESTINATION lib/${PROJECT_NAME})
    endif()
  endif()

  # OKVIS2X nodes - Wo/ Deep Learning W/Deep Learning
  if(USE_NN)
    # W/ DFusion Processor
    add_executable(okvis2x_depthfusion_network_node_subscriber okvis_ros2/src/okvis2x_network_node.cpp)
    target_compile_definitions(okvis2x_depthfusion_network_node_subscriber PUBLIC OKVIS_DFUSION_NETWORK_PROCESSOR)
    target_link_libraries(
      okvis2x_depthfusion_network_node_subscriber
      okvis::Ros2
      ${OpenCV_LIBS}
      ${SUITESPARSE_LIBRARIES}
      ${Boost_LIBRARIES} 
      ${PCL_LIBRARIES}
      okvis::DeepLearning
    )
    install(TARGETS okvis2x_depthfusion_network_node_subscriber DESTINATION lib/${PROJECT_NAME})

    if(USE_COLIDMAP)
      # W/ Vision-Language Mapping (FindAnything)
      add_executable(okvis2x_language_network_node_subscriber okvis_ros2/src/okvis2x_network_node.cpp)
      target_compile_definitions(okvis2x_language_network_node_subscriber PUBLIC OKVIS_LANGUAGE_NETWORK_PROCESSOR)
      target_link_libraries(
        okvis2x_language_network_node_subscriber
        okvis::Ros2
        ${OpenCV_LIBS}
        ${SUITESPARSE_LIBRARIES}
        ${Boost_LIBRARIES} 
        ${PCL_LIBRARIES}
        okvis::DeepLearning
      )
      ament_target_dependencies(okvis2x_language_network_node_subscriber std_msgs language_feature_msgs)
      install(TARGETS okvis2x_language_network_node_subscriber DESTINATION lib/${PROJECT_NAME})

      # W/ Stereo Network Processor AND Vision-Language Mapping (FindAnything): the stereo network
      # provides the depth, so this needs no depth camera.
      add_executable(okvis2x_stereo_language_node_subscriber okvis_ros2/src/okvis2x_network_node.cpp)
      target_compile_definitions(okvis2x_stereo_language_node_subscriber
                                 PUBLIC OKVIS_STEREO_NETWORK_PROCESSOR OKVIS_LANGUAGE_NETWORK_PROCESSOR)
      target_link_libraries(
        okvis2x_stereo_language_node_subscriber
        okvis::Ros2
        ${OpenCV_LIBS}
        ${SUITESPARSE_LIBRARIES}
        ${Boost_LIBRARIES}
        ${PCL_LIBRARIES}
        okvis::DeepLearning
      )
      ament_target_dependencies(okvis2x_stereo_language_node_subscriber std_msgs language_feature_msgs)
      install(TARGETS okvis2x_stereo_language_node_subscriber DESTINATION lib/${PROJECT_NAME})
    endif()

    if(HAVE_LIBREALSENSE)
      if(USE_COLIDMAP)
        # W/ Vision-Language Mapping (FindAnything)
        add_executable(okvis2x_language_network_node_realsense okvis_ros2/src/okvis2x_network_node.cpp)
        target_compile_definitions(okvis2x_language_network_node_realsense PUBLIC OKVIS_LANGUAGE_NETWORK_PROCESSOR SRL_NAV_USE_REALSENSE)
        target_link_libraries(
          okvis2x_language_network_node_realsense
          okvis::Ros2
          ${OpenCV_LIBS}
          ${SUITESPARSE_LIBRARIES}
          ${Boost_LIBRARIES} 
          ${PCL_LIBRARIES}
          okvis::DeepLearning
        )
        ament_target_dependencies(okvis2x_language_network_node_realsense std_msgs language_feature_msgs)
        install(TARGETS okvis2x_language_network_node_realsense DESTINATION lib/${PROJECT_NAME})
      endif()

      # W/ DepthFusion
      add_executable(okvis2x_depthfusion_network_node_realsense okvis_ros2/src/okvis2x_network_node.cpp)
      target_compile_definitions(okvis2x_depthfusion_network_node_realsense PUBLIC OKVIS_DFUSION_NETWORK_PROCESSOR SRL_NAV_USE_REALSENSE)
      target_link_libraries(
        okvis2x_depthfusion_network_node_realsense
        okvis::Ros2
        ${OpenCV_LIBS}
        ${SUITESPARSE_LIBRARIES}
        ${Boost_LIBRARIES} 
        ${PCL_LIBRARIES}
        okvis::DeepLearning
      )
      install(TARGETS okvis2x_depthfusion_network_node_realsense DESTINATION lib/${PROJECT_NAME})

      # W/ StereoNetwork
      add_executable(okvis2x_stereo_network_node_realsense okvis_ros2/src/okvis2x_network_node.cpp)
      target_compile_definitions(okvis2x_stereo_network_node_realsense PUBLIC OKVIS_STEREO_NETWORK_PROCESSOR SRL_NAV_USE_REALSENSE)
      target_link_libraries(
        okvis2x_stereo_network_node_realsense
        okvis::Ros2
        ${OpenCV_LIBS}
        ${SUITESPARSE_LIBRARIES}
        ${Boost_LIBRARIES} 
        ${PCL_LIBRARIES}
        okvis::DeepLearning
      )
      install(TARGETS okvis2x_stereo_network_node_realsense DESTINATION lib/${PROJECT_NAME})
    endif()

  endif()

  # Install launch files
  install(DIRECTORY okvis_ros2/launch DESTINATION share/${PROJECT_NAME}/)
  install(DIRECTORY resources DESTINATION share/${PROJECT_NAME}/)
  install(DIRECTORY config DESTINATION share/${PROJECT_NAME}/)

  # Form ROS2 Package
  AMENT_PACKAGE()

endif(BUILD_ROS2)

# The directory where the .cmake files will be installed
include(GNUInstallDirs)
if(WIN32 AND NOT CYGWIN)
    set(INSTALL_CMAKEDIR "${PACKAGE_NAME}/cmake")
elseif(APPLE)
    set(INSTALL_CMAKEDIR "${PACKAGE_NAME}.framework/Resources/CMake")
else()
    set(INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}")
endif()
# Generate okvisConfig.cmake
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/okvisConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/okvisConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CMAKEDIR})
# Generate okvisConfigVersion.cmake
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/okvisConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)
install(EXPORT okvisTargets
  FILE okvisTargets.cmake
  NAMESPACE okvis::
  DESTINATION ${INSTALL_CMAKEDIR}/okvis)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/okvisConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/okvisConfigVersion.cmake
  DESTINATION ${INSTALL_CMAKEDIR}/okvis)