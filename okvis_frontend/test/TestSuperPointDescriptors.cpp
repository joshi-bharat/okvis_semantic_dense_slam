/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#include <gtest/gtest.h>
#include <okvis/SuperPointDescriptors.hpp>

namespace {
// A 2-dimensional descriptor grid, 2x2 cells, so bilinear weights are easy to check by hand.
cv::Mat makeCoarse() {
  cv::Mat coarse(2, 4, CV_32F); // D = 2 rows, Hc*Wc = 4
  coarse.at<float>(0, 0) = 1.0f; coarse.at<float>(0, 1) = 3.0f; // dim 0: [1 3; 5 7]
  coarse.at<float>(0, 2) = 5.0f; coarse.at<float>(0, 3) = 7.0f;
  coarse.at<float>(1, 0) = 0.0f; coarse.at<float>(1, 1) = 0.0f; // dim 1: all zero
  coarse.at<float>(1, 2) = 0.0f; coarse.at<float>(1, 3) = 0.0f;
  return coarse;
}
} // namespace

TEST(SuperPointDescriptors, bilinearlySamplesTheCoarseGrid) {
  // Score map 16x16 over a 2x2 grid -> 8 px per cell, cell centres at 4 and 12.
  // A keypoint at (8, 4) sits between columns 0 and 1 of row 0.
  const cv::Mat coarse = makeCoarse();
  const std::vector<cv::KeyPoint> keypoints{cv::KeyPoint(cv::Point2f(8.0f, 4.0f), 1.0f)};
  const cv::Mat descriptors =
      okvis::sampleSuperPointDescriptors(coarse, keypoints, cv::Size(16, 16));
  ASSERT_EQ(descriptors.rows, 1);
  ASSERT_EQ(descriptors.cols, 2);
  ASSERT_EQ(descriptors.type(), CV_16F);
  cv::Mat asFloat;
  descriptors.convertTo(asFloat, CV_32F);
  // Only dimension 0 is non-zero, so after L2 normalisation it must be exactly 1.
  EXPECT_NEAR(asFloat.at<float>(0, 0), 1.0f, 1e-3f);
  EXPECT_NEAR(asFloat.at<float>(0, 1), 0.0f, 1e-3f);
}

TEST(SuperPointDescriptors, interpolatesWithHandComputedWeights) {
  // Dim 0 is [1 3; 5 7], dim 1 is constant 1, so the normalised ratio dim0/dim1 is the
  // interpolated dim-0 value. Pixel centres are at x + 0.5, so (7.5, 3.5) maps to grid
  // (0.5, 0.0): halfway between columns 0 and 1 of row 0 -> (1 + 3) / 2 = 2.
  cv::Mat coarse = makeCoarse();
  coarse.row(1).setTo(1.0f);
  const std::vector<cv::KeyPoint> keypoints{cv::KeyPoint(cv::Point2f(7.5f, 3.5f), 1.0f)};
  const cv::Mat descriptors =
      okvis::sampleSuperPointDescriptors(coarse, keypoints, cv::Size(16, 16));
  cv::Mat asFloat;
  descriptors.convertTo(asFloat, CV_32F);
  const float norm = std::sqrt(5.0f); // |(2, 1)|
  EXPECT_NEAR(asFloat.at<float>(0, 0), 2.0f / norm, 1e-3f);
  EXPECT_NEAR(asFloat.at<float>(0, 1), 1.0f / norm, 1e-3f);
}

TEST(SuperPointDescriptors, normalisesEachRowToUnitLength) {
  cv::Mat coarse(2, 4, CV_32F);
  coarse.at<float>(0, 0) = 3.0f; coarse.at<float>(0, 1) = 3.0f;
  coarse.at<float>(0, 2) = 3.0f; coarse.at<float>(0, 3) = 3.0f;
  coarse.at<float>(1, 0) = 4.0f; coarse.at<float>(1, 1) = 4.0f;
  coarse.at<float>(1, 2) = 4.0f; coarse.at<float>(1, 3) = 4.0f;
  const std::vector<cv::KeyPoint> keypoints{cv::KeyPoint(cv::Point2f(4.0f, 4.0f), 1.0f)};
  const cv::Mat descriptors =
      okvis::sampleSuperPointDescriptors(coarse, keypoints, cv::Size(16, 16));
  cv::Mat asFloat;
  descriptors.convertTo(asFloat, CV_32F);
  EXPECT_NEAR(asFloat.at<float>(0, 0), 0.6f, 1e-3f); // 3 / 5
  EXPECT_NEAR(asFloat.at<float>(0, 1), 0.8f, 1e-3f); // 4 / 5
  const double norm = cv::norm(asFloat.row(0), cv::NORM_L2);
  EXPECT_NEAR(norm, 1.0, 1e-3);
}

TEST(SuperPointDescriptors, returnsEmptyForNoKeypoints) {
  const cv::Mat descriptors =
      okvis::sampleSuperPointDescriptors(makeCoarse(), {}, cv::Size(16, 16));
  EXPECT_EQ(descriptors.rows, 0);
}
