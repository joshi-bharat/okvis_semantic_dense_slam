/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>
#include <okvis/ViParametersReader.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <stdexcept>

namespace {

/**
 * \brief Copy a real config into a temporary directory, optionally injecting extra lines into
 *        frontend_parameters.
 *
 * The reader insists on the full set of camera, IMU and estimator entries, so a hand-written
 * stub is not enough. Writing into a temporary directory also lets us check that a relative
 * engine path resolves against the config's own directory rather than the repository's.
 */
std::string writeConfig(const std::string & name, const std::string & extraFrontendLines,
                        const std::string & anchor = "frontend_parameters:") {
  const std::filesystem::path base =
      std::filesystem::path(OKVIS_TEST_CONFIG_DIR) / "uhumans2" / "okvis2.yaml";
  std::ifstream in(base);
  EXPECT_TRUE(in.good()) << "cannot read " << base;
  std::stringstream buffer;
  buffer << in.rdbuf();
  std::string text = buffer.str();

  if (!extraFrontendLines.empty()) {
    const size_t anchorPosition = text.find(anchor);
    EXPECT_NE(anchorPosition, std::string::npos) << "no " << anchor << " in " << base;
    const size_t endOfLine = text.find('\n', anchorPosition);
    text.insert(endOfLine + 1, extraFrontendLines);
  }

  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "okvis_superpoint_test";
  std::filesystem::create_directories(dir);
  const std::filesystem::path file = dir / name;
  std::ofstream out(file);
  out << text;
  return file.string();
}

} // namespace

TEST(SuperPointParameters, defaultsToBriskWhenAbsent) {
  const std::string file = writeConfig("brisk.yaml", "");
  okvis::ViParametersReader reader(file);
  okvis::ViParameters parameters;
  reader.getParameters(parameters);
  EXPECT_TRUE(parameters.superpoint.engine.empty());
  // Defaults must still be usable, so a later engine only needs the path.
  EXPECT_DOUBLE_EQ(parameters.superpoint.threshold, 0.005);
  EXPECT_EQ(parameters.superpoint.nms_radius, 4);
  EXPECT_DOUBLE_EQ(parameters.superpoint.matching_threshold, 0.8);
}

TEST(SuperPointParameters, readsValuesAndResolvesRelativeEnginePath) {
  const std::string file = writeConfig("superpoint.yaml",
                                       "    superpoint_engine: superpoint.engine\n"
                                       "    superpoint_threshold: 0.01\n"
                                       "    superpoint_nms_radius: 3\n"
                                       "    superpoint_matching_threshold: 0.75\n");
  okvis::ViParametersReader reader(file);
  okvis::ViParameters parameters;
  reader.getParameters(parameters);
  const std::filesystem::path expected =
      std::filesystem::path(file).parent_path() / "superpoint.engine";
  EXPECT_EQ(parameters.superpoint.engine, expected.lexically_normal().string());
  EXPECT_DOUBLE_EQ(parameters.superpoint.threshold, 0.01);
  EXPECT_EQ(parameters.superpoint.nms_radius, 3);
  EXPECT_DOUBLE_EQ(parameters.superpoint.matching_threshold, 0.75);
}

TEST(SuperPointParameters, keepsAbsoluteEnginePathUnchanged) {
  const std::string file =
      writeConfig("absolute.yaml", "    superpoint_engine: /opt/models/superpoint.engine\n");
  okvis::ViParametersReader reader(file);
  okvis::ViParameters parameters;
  reader.getParameters(parameters);
  EXPECT_EQ(parameters.superpoint.engine, "/opt/models/superpoint.engine");
}

TEST(SuperPointParameters, readsItsOwnKeypointBudget) {
  const std::string file =
      writeConfig("budget.yaml", "    superpoint_max_num_keypoints: 1000\n");
  okvis::ViParametersReader reader(file);
  okvis::ViParameters parameters;
  reader.getParameters(parameters);
  EXPECT_EQ(parameters.superpoint.max_num_keypoints, 1000);
}

TEST(SuperPointParameters, keypointBudgetDefaultsToSharedOne) {
  const std::string file = writeConfig("shared-budget.yaml", "");
  okvis::ViParametersReader reader(file);
  okvis::ViParameters parameters;
  reader.getParameters(parameters);
  EXPECT_EQ(parameters.superpoint.max_num_keypoints, 0); // 0: use max_num_keypoints
}

// image_size resizes every camera; the uhumans2 cameras are 720x480 with f = 415.69, c = (360, 240).
TEST(ImageResize, scalesIntrinsicsToTheNewImageSize) {
  const std::string file =
      writeConfig("resize.yaml", "    image_size: [360, 240]\n", "camera_parameters:");
  okvis::ViParametersReader reader(file);
  okvis::ViParameters parameters;
  reader.getParameters(parameters);
  EXPECT_EQ(parameters.camera.image_size[0], 360);
  EXPECT_EQ(parameters.camera.image_size[1], 240);
  for (size_t i = 0; i < parameters.nCameraSystem.numCameras(); ++i) {
    const auto geometry = parameters.nCameraSystem.cameraGeometry(i);
    EXPECT_EQ(geometry->imageWidth(), 360u);
    EXPECT_EQ(geometry->imageHeight(), 240u);
    Eigen::VectorXd intrinsics;
    geometry->getIntrinsics(intrinsics);
    EXPECT_NEAR(intrinsics[0], 415.69219381653056 * 0.5, 1e-9); // fu
    EXPECT_NEAR(intrinsics[1], 415.69219381653056 * 0.5, 1e-9); // fv
    // Pixel centres at x + 0.5, as cv::resize uses: (360 + 0.5) * 0.5 - 0.5.
    EXPECT_NEAR(intrinsics[2], 179.75, 1e-9); // cu
    EXPECT_NEAR(intrinsics[3], 119.75, 1e-9); // cv
  }
}

TEST(ImageResize, keepsIntrinsicsWithoutImageSize) {
  const std::string file = writeConfig("noresize.yaml", "");
  okvis::ViParametersReader reader(file);
  okvis::ViParameters parameters;
  reader.getParameters(parameters);
  EXPECT_EQ(parameters.camera.image_size[0], 0);
  Eigen::VectorXd intrinsics;
  parameters.nCameraSystem.cameraGeometry(0)->getIntrinsics(intrinsics);
  EXPECT_NEAR(intrinsics[0], 415.69219381653056, 1e-9);
  EXPECT_NEAR(intrinsics[2], 360.0, 1e-9);
}

TEST(ImageResize, rejectsAChangedAspectRatio) {
  const std::string file =
      writeConfig("badresize.yaml", "    image_size: [360, 360]\n", "camera_parameters:");
  EXPECT_THROW(okvis::ViParametersReader reader(file), std::invalid_argument);
}
