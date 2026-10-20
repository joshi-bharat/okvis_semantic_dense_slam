/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#include <gtest/gtest.h>
#include <okvis/SuperPointSparse.hpp>

namespace {
// Keypoints as an engine with built-in top-k returns them: N x 2 (x, y), N scores, N x D descriptors.
struct SparseOutput {
  cv::Mat keypoints; // CV_32F, N x 2
  cv::Mat scores;    // CV_32F, 1 x N
  cv::Mat descriptors; // CV_32F, N x D
};

SparseOutput makeOutput(const std::vector<cv::Point2f> & points, const std::vector<float> & scores,
                        int dimension = 4) {
  SparseOutput output;
  output.keypoints = cv::Mat(int(points.size()), 2, CV_32F);
  output.scores = cv::Mat(1, int(points.size()), CV_32F);
  output.descriptors = cv::Mat::zeros(int(points.size()), dimension, CV_32F);
  for (int i = 0; i < int(points.size()); ++i) {
    output.keypoints.at<float>(i, 0) = points[size_t(i)].x;
    output.keypoints.at<float>(i, 1) = points[size_t(i)].y;
    output.scores.at<float>(0, i) = scores[size_t(i)];
    output.descriptors.at<float>(i, i % dimension) = 2.0f; // unnormalised on purpose
  }
  return output;
}
} // namespace

TEST(SuperPointSparse, dropsKeypointsBelowThreshold) {
  const SparseOutput out = makeOutput({{20, 20}, {40, 40}}, {0.5f, 0.001f});
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  okvis::selectSparseSuperPoint(out.keypoints, out.scores, out.descriptors, 0.005f, 4, 100,
                                keypoints, descriptors);
  ASSERT_EQ(keypoints.size(), 1u);
  EXPECT_FLOAT_EQ(keypoints[0].pt.x, 20.0f);
}

TEST(SuperPointSparse, suppressesWeakerKeypointsWithinRadius) {
  // 5 px apart: radius 8 keeps only the stronger one, radius 4 keeps both.
  const SparseOutput out = makeOutput({{20, 20}, {25, 20}}, {0.3f, 0.6f});
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  okvis::selectSparseSuperPoint(out.keypoints, out.scores, out.descriptors, 0.005f, 8, 100,
                                keypoints, descriptors);
  ASSERT_EQ(keypoints.size(), 1u);
  EXPECT_FLOAT_EQ(keypoints[0].pt.x, 25.0f);
  EXPECT_FLOAT_EQ(keypoints[0].response, 0.6f);

  okvis::selectSparseSuperPoint(out.keypoints, out.scores, out.descriptors, 0.005f, 4, 100,
                                keypoints, descriptors);
  EXPECT_EQ(keypoints.size(), 2u);
}

TEST(SuperPointSparse, honoursTheKeypointBudgetStrongestFirst) {
  const SparseOutput out = makeOutput({{10, 10}, {30, 10}, {50, 10}}, {0.2f, 0.9f, 0.5f});
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  okvis::selectSparseSuperPoint(out.keypoints, out.scores, out.descriptors, 0.005f, 4, 2,
                                keypoints, descriptors);
  ASSERT_EQ(keypoints.size(), 2u);
  EXPECT_FLOAT_EQ(keypoints[0].response, 0.9f);
  EXPECT_FLOAT_EQ(keypoints[1].response, 0.5f);
  EXPECT_EQ(descriptors.rows, 2);
}

TEST(SuperPointSparse, returnsNormalisedFp16DescriptorsInKeypointOrder) {
  // Keypoint 1 (x = 30) is strongest, so its descriptor (non-zero in dimension 1) comes first.
  const SparseOutput out = makeOutput({{10, 10}, {30, 10}}, {0.2f, 0.9f});
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  okvis::selectSparseSuperPoint(out.keypoints, out.scores, out.descriptors, 0.005f, 4, 100,
                                keypoints, descriptors);
  ASSERT_EQ(descriptors.type(), CV_16F);
  ASSERT_EQ(descriptors.rows, 2);
  cv::Mat asFloat;
  descriptors.convertTo(asFloat, CV_32F);
  EXPECT_NEAR(asFloat.at<float>(0, 1), 1.0f, 1e-3f);
  EXPECT_NEAR(asFloat.at<float>(1, 0), 1.0f, 1e-3f);
  EXPECT_NEAR(cv::norm(asFloat.row(0), cv::NORM_L2), 1.0, 1e-3);
}
