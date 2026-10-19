/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#include <gtest/gtest.h>
#include <okvis/CublasMatcher.hpp>

namespace {
cv::Mat randomNormalisedDescriptors(int rows, int dimension, int seed) {
  cv::RNG rng{uint64_t(seed)};
  cv::Mat descriptors(rows, dimension, CV_32F);
  rng.fill(descriptors, cv::RNG::NORMAL, 0.0, 1.0);
  for (int r = 0; r < rows; ++r) {
    descriptors.row(r) /= cv::norm(descriptors.row(r), cv::NORM_L2);
  }
  cv::Mat half;
  descriptors.convertTo(half, CV_16F);
  return half;
}

cv::Mat referenceScores(const cv::Mat & query, const cv::Mat & train) {
  cv::Mat q, t;
  query.convertTo(q, CV_32F);
  train.convertTo(t, CV_32F);
  return q * t.t();
}
} // namespace

TEST(CublasMatcher, matchesACpuReferenceWithinFp16Tolerance) {
  const cv::Mat query = randomNormalisedDescriptors(64, 256, 1);
  const cv::Mat train = randomNormalisedDescriptors(128, 256, 2);
  okvis::CublasMatcher matcher;
  const cv::Mat scores = matcher.score(query, train);
  ASSERT_EQ(scores.rows, 64);
  ASSERT_EQ(scores.cols, 128);
  ASSERT_EQ(scores.type(), CV_32F);
  const cv::Mat expected = referenceScores(query, train);
  double maxError = 0.0;
  for (int i = 0; i < scores.rows; ++i) {
    for (int j = 0; j < scores.cols; ++j) {
      maxError = std::max(maxError,
                          std::abs(double(scores.at<float>(i, j) - expected.at<float>(i, j))));
    }
  }
  EXPECT_LT(maxError, 5e-3);
}

TEST(CublasMatcher, scoresIdenticalDescriptorsAsOne) {
  const cv::Mat descriptors = randomNormalisedDescriptors(8, 256, 3);
  okvis::CublasMatcher matcher;
  const cv::Mat scores = matcher.score(descriptors, descriptors);
  for (int i = 0; i < scores.rows; ++i) {
    EXPECT_NEAR(scores.at<float>(i, i), 1.0f, 5e-3f);
  }
}

TEST(CublasMatcher, handlesEmptyInput) {
  okvis::CublasMatcher matcher;
  const cv::Mat scores = matcher.score(cv::Mat(0, 256, CV_16F),
                                       randomNormalisedDescriptors(4, 256, 4));
  EXPECT_EQ(scores.rows, 0);
}
