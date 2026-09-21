/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#include <gtest/gtest.h>
#include <okvis/SuperPointNms.hpp>

TEST(SuperPointNms, keepsIsolatedPeaksAboveThreshold) {
  cv::Mat scores = cv::Mat::zeros(40, 40, CV_32F);
  scores.at<float>(10, 10) = 0.9f;
  scores.at<float>(30, 30) = 0.5f;
  scores.at<float>(20, 20) = 0.001f; // below threshold
  const auto keypoints = okvis::superPointNms(scores, 0.01f, 4, 10);
  ASSERT_EQ(keypoints.size(), 2u);
  EXPECT_FLOAT_EQ(keypoints[0].response, 0.9f); // strongest first
  EXPECT_FLOAT_EQ(keypoints[0].pt.x, 10.0f);
  EXPECT_FLOAT_EQ(keypoints[0].pt.y, 10.0f);
  EXPECT_FLOAT_EQ(keypoints[1].response, 0.5f);
}

TEST(SuperPointNms, suppressesWeakerNeighbourWithinRadius) {
  cv::Mat scores = cv::Mat::zeros(40, 40, CV_32F);
  scores.at<float>(20, 20) = 0.9f;
  scores.at<float>(20, 22) = 0.8f; // 2 px from the 0.9 peak, radius 4 -> suppressed
  // Clear of both: 10 px from the 0.9 peak and 8 px from the 0.8 one. Suppression does not
  // chain, so being outside the radius of a suppressed pixel is not enough - it has to be
  // outside the radius of every stronger pixel, as with a max-pool.
  scores.at<float>(20, 30) = 0.7f;
  const auto keypoints = okvis::superPointNms(scores, 0.01f, 4, 10);
  ASSERT_EQ(keypoints.size(), 2u);
  EXPECT_FLOAT_EQ(keypoints[0].response, 0.9f);
  EXPECT_FLOAT_EQ(keypoints[1].response, 0.7f);
}

TEST(SuperPointNms, dropsPeaksInsideTheBorderMargin) {
  cv::Mat scores = cv::Mat::zeros(40, 40, CV_32F);
  scores.at<float>(2, 20) = 0.9f;  // within radius 4 of the top border
  scores.at<float>(20, 37) = 0.9f; // within radius 4 of the right border
  scores.at<float>(20, 20) = 0.3f;
  const auto keypoints = okvis::superPointNms(scores, 0.01f, 4, 10);
  ASSERT_EQ(keypoints.size(), 1u);
  EXPECT_FLOAT_EQ(keypoints[0].pt.x, 20.0f);
  EXPECT_FLOAT_EQ(keypoints[0].pt.y, 20.0f);
}

TEST(SuperPointNms, honoursTheKeypointBudget) {
  cv::Mat scores = cv::Mat::zeros(100, 100, CV_32F);
  for (int i = 0; i < 10; ++i) {
    scores.at<float>(10 + 8 * i, 50) = 0.1f * float(i + 1);
  }
  const auto keypoints = okvis::superPointNms(scores, 0.01f, 4, 3);
  ASSERT_EQ(keypoints.size(), 3u);
  EXPECT_GT(keypoints[0].response, keypoints[1].response);
  EXPECT_GT(keypoints[1].response, keypoints[2].response);
}

TEST(SuperPointNms, breaksTiesDeterministically) {
  cv::Mat scores = cv::Mat::zeros(60, 60, CV_32F);
  scores.at<float>(20, 40) = 0.5f;
  scores.at<float>(20, 20) = 0.5f;
  scores.at<float>(40, 20) = 0.5f;
  const auto first = okvis::superPointNms(scores, 0.01f, 4, 10);
  const auto second = okvis::superPointNms(scores, 0.01f, 4, 10);
  ASSERT_EQ(first.size(), 3u);
  for (size_t i = 0; i < first.size(); ++i) {
    EXPECT_FLOAT_EQ(first[i].pt.x, second[i].pt.x);
    EXPECT_FLOAT_EQ(first[i].pt.y, second[i].pt.y);
  }
  // Equal scores: lower row first, then lower column.
  EXPECT_FLOAT_EQ(first[0].pt.y, 20.0f);
  EXPECT_FLOAT_EQ(first[0].pt.x, 20.0f);
}

TEST(SuperPointNms, keepsAdjacentEqualScoresFromBothSurviving) {
  // Two touching pixels of identical score: exactly one must survive, or a cluster of equal
  // scores would produce a keypoint per pixel.
  cv::Mat scores = cv::Mat::zeros(40, 40, CV_32F);
  scores.at<float>(20, 20) = 0.5f;
  scores.at<float>(20, 21) = 0.5f;
  const auto keypoints = okvis::superPointNms(scores, 0.01f, 4, 10);
  ASSERT_EQ(keypoints.size(), 1u);
  EXPECT_FLOAT_EQ(keypoints[0].pt.x, 20.0f);
}

TEST(SuperPointNms, returnsNothingForAnEmptyScoreMap) {
  const cv::Mat scores = cv::Mat::zeros(40, 40, CV_32F);
  EXPECT_TRUE(okvis::superPointNms(scores, 0.01f, 4, 10).empty());
}
