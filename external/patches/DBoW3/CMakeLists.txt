cmake_minimum_required(VERSION 3.8...3.16)
project(DBoW3)

# Replaces DBoW3's own CMakeLists.txt: a static library with the headers exposed as <DBoW3/...>
# (they live flat in src/ upstream, with generic names such as Database.h).

find_package(OpenCV REQUIRED COMPONENTS core)

set(DBOW3_HEADERS
  src/BowVector.h src/Database.h src/DBoW3.h src/DescManip.h src/exports.h src/FeatureVector.h
  src/QueryResults.h src/quicklz.h src/ScoringObject.h src/timers.h src/Vocabulary.h)
set(DBOW3_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${DBOW3_INCLUDE_DIR}/DBoW3)
foreach(header ${DBOW3_HEADERS})
  configure_file(${header} ${DBOW3_INCLUDE_DIR}/DBoW3/ COPYONLY)
endforeach()

add_library(${PROJECT_NAME}
  src/BowVector.cpp
  src/Database.cpp
  src/DescManip.cpp
  src/FeatureVector.cpp
  src/QueryResults.cpp
  src/quicklz.c
  src/ScoringObject.cpp
  src/Vocabulary.cpp
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${DBOW3_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME} EXPORT DBoW3Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY ${DBOW3_INCLUDE_DIR}/DBoW3 DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT DBoW3Targets
  FILE DBoW3Targets.cmake
  NAMESPACE DBoW3::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DBoW3)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/DBoW3Config.cmake
  "include(CMakeFindDependencyMacro)\nfind_dependency(OpenCV)\n"
  "include(\"\${CMAKE_CURRENT_LIST_DIR}/DBoW3Targets.cmake\")\n")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/DBoW3Config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DBoW3)
