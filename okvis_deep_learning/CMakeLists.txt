cmake_minimum_required(VERSION 3.5)
project(okvis_deep_learning)

find_package(Eigen3 REQUIRED)
if(USE_NN AND NOT Torch_FOUND)
    find_package(Torch REQUIRED)
endif()
if(USE_NN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()
find_package(OpenCV COMPONENTS core imgproc features2d calib3d dnn REQUIRED)

set(LIB_NAME okvis_deep_learning)

add_library(${LIB_NAME} src/Processor.cpp)
if(USE_NN)
    target_sources(${LIB_NAME}
        PRIVATE
            src/Stereo2DepthProcessor.cpp
            src/VisionLanguageProcessor.cpp
            src/DepthFusionProcessor.cpp)
endif()

# TensorRT stereo depth backend (no libtorch needed); the engine wrapper is okvis::TensorRt
if(USE_TENSORRT)
    target_sources(${LIB_NAME}
        PRIVATE
            src/TensorRtStereo2DepthProcessor.cpp)
    target_link_libraries(${LIB_NAME} okvis::TensorRt)
endif()

target_include_directories(${LIB_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
    ${Torch_INCLUDE_DIRS}
    )

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
    okvis::Ceres
    okvis::Common
    okvis::CV
    okvis::Frontend
    okvis::Kinematics
    okvis::Mapping
    okvis::MultisensorProcessing
    okvis::Time
    okvis::Timing
    okvis::Util
)

if(USE_NN)
    target_link_libraries(${PROJECT_NAME}
        ${TORCH_LIBRARIES}
    )
endif()


add_library(okvis::DeepLearning ALIAS ${LIB_NAME})
install(TARGETS ${LIB_NAME} EXPORT okvisTargets
  ARCHIVE COMPONENT lib
  LIBRARY COMPONENT lib
)
install(DIRECTORY include/okvis TYPE INCLUDE COMPONENT dev FILES_MATCHING PATTERN "*.hpp")
