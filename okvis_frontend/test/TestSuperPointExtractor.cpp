/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#include <gtest/gtest.h>
#include <okvis/SuperPointExtractor.hpp>

namespace {
// Stands in for the network: one peak at a known location in a 64x64 score map, and a
// constant descriptor grid so the descriptor values are predictable.
void fakeInference(const cv::Mat & input, cv::Mat & scores, cv::Mat & coarse) {
  EXPECT_EQ(input.cols, 64);
  EXPECT_EQ(input.rows, 64);
  scores = cv::Mat::zeros(64, 64, CV_32F);
  scores.at<float>(32, 16) = 0.9f; // row 32, column 16
  coarse = cv::Mat::ones(4, 64, CV_32F); // D = 4, 8x8 grid
}
} // namespace

TEST(SuperPointExtractor, mapsKeypointsBackToImageResolution) {
  okvis::SuperPointParameters parameters;
  parameters.threshold = 0.01;
  parameters.nms_radius = 4;
  okvis::SuperPointExtractor extractor(parameters, 100, cv::Size(64, 64), fakeInference);

  // A 128x128 image is downscaled by 2 for the network. With pixel centres at x + 0.5, as
  // cv::resize uses, score-map pixel (16, 32) maps back to (32.5, 64.5).
  const cv::Mat image = cv::Mat::zeros(128, 128, CV_8UC1);
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  extractor.extract(image, keypoints, descriptors);

  ASSERT_EQ(keypoints.size(), 1u);
  EXPECT_NEAR(keypoints[0].pt.x, 32.5f, 1e-3f);
  EXPECT_NEAR(keypoints[0].pt.y, 64.5f, 1e-3f);
  EXPECT_FLOAT_EQ(keypoints[0].response, 0.9f);
}

TEST(SuperPointExtractor, producesNormalisedFp16Descriptors) {
  okvis::SuperPointParameters parameters;
  parameters.threshold = 0.01;
  parameters.nms_radius = 4;
  okvis::SuperPointExtractor extractor(parameters, 100, cv::Size(64, 64), fakeInference);

  const cv::Mat image = cv::Mat::zeros(128, 128, CV_8UC1);
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  extractor.extract(image, keypoints, descriptors);

  ASSERT_EQ(descriptors.rows, 1);
  ASSERT_EQ(descriptors.cols, 4);
  ASSERT_EQ(descriptors.type(), CV_16F);
  cv::Mat asFloat;
  descriptors.convertTo(asFloat, CV_32F);
  EXPECT_NEAR(cv::norm(asFloat.row(0), cv::NORM_L2), 1.0, 1e-3);
  EXPECT_NEAR(asFloat.at<float>(0, 0), 0.5f, 1e-3f); // constant grid, D = 4 -> 1/sqrt(4)
}

TEST(SuperPointExtractor, respectsTheKeypointBudget) {
  okvis::SuperPointParameters parameters;
  parameters.threshold = 0.01;
  parameters.nms_radius = 2;
  okvis::SuperPointExtractor extractor(
      parameters, 2, cv::Size(64, 64),
      [](const cv::Mat &, cv::Mat & scores, cv::Mat & coarse) {
        scores = cv::Mat::zeros(64, 64, CV_32F);
        scores.at<float>(20, 20) = 0.9f;
        scores.at<float>(30, 30) = 0.8f;
        scores.at<float>(40, 40) = 0.7f;
        coarse = cv::Mat::ones(4, 64, CV_32F);
      });
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  extractor.extract(cv::Mat::zeros(64, 64, CV_8UC1), keypoints, descriptors);
  EXPECT_EQ(keypoints.size(), 2u);
  EXPECT_EQ(descriptors.rows, 2);
}
