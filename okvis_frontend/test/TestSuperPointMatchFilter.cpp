/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#include <limits>

#include <gtest/gtest.h>
#include <okvis/SuperPointMatchFilter.hpp>

namespace {
const float kInf = std::numeric_limits<float>::infinity();

cv::Mat matrix(std::initializer_list<std::initializer_list<float>> rows) {
  cv::Mat m(int(rows.size()), int(rows.begin()->size()), CV_32F);
  int r = 0;
  for (const auto & row : rows) {
    int c = 0;
    for (const float v : row) {
      m.at<float>(r, c++) = v;
    }
    ++r;
  }
  return m;
}

// One candidate per column, every pair admissible.
std::vector<std::pair<int, int>> columns(int n) {
  std::vector<std::pair<int, int>> groups;
  for (int c = 0; c < n; ++c) {
    groups.emplace_back(c, c + 1);
  }
  return groups;
}

bool anyPair(int, int) {
  return true;
}
} // namespace

TEST(SuperPointMatchFilter, ratioTestRejectsAmbiguousMatches) {
  // Query 0: 0.10 vs 0.11 -> ambiguous. Query 1: 0.05 vs 0.80 -> distinctive.
  const cv::Mat d = matrix({{0.10f, 0.11f, 1.0f}, {1.0f, 0.05f, 0.80f}});
  const cv::Mat out = okvis::filterSuperPointMatches(d, columns(3), anyPair, 0.8, false);
  EXPECT_EQ(out.at<float>(0, 0), kInf);
  EXPECT_EQ(out.at<float>(0, 1), kInf);
  EXPECT_FLOAT_EQ(out.at<float>(1, 1), 0.05f);
  EXPECT_EQ(out.at<float>(1, 2), kInf); // only the winner survives
}

TEST(SuperPointMatchFilter, ratioIsOnL2DistanceNotCosine) {
  // d = 1 - cos, L2 = sqrt(2 d): 0.25 vs 0.36 -> L2 ratio sqrt(0.25 / 0.36) = 0.833.
  const cv::Mat d = matrix({{0.25f, 0.36f}});
  EXPECT_FLOAT_EQ(okvis::filterSuperPointMatches(d, columns(2), anyPair, 0.85, false)
                      .at<float>(0, 0), 0.25f);
  EXPECT_EQ(okvis::filterSuperPointMatches(d, columns(2), anyPair, 0.80, false).at<float>(0, 0),
            kInf);
}

TEST(SuperPointMatchFilter, crossCheckKeepsOnlyMutualBestMatches) {
  // Both queries prefer candidate 0, but candidate 0 prefers query 1.
  const cv::Mat d = matrix({{0.20f, 0.90f}, {0.10f, 0.90f}});
  const cv::Mat out = okvis::filterSuperPointMatches(d, columns(2), anyPair, 1.0, true);
  EXPECT_EQ(out.at<float>(0, 0), kInf);
  EXPECT_FLOAT_EQ(out.at<float>(1, 0), 0.10f);
}

TEST(SuperPointMatchFilter, groupsAreOneCandidateWithSeveralDescriptors) {
  // Candidate 0 owns columns 0-1 (two stored descriptors), candidate 1 owns column 2. The two
  // descriptors of candidate 0 must not count as each other's second best.
  const cv::Mat d = matrix({{0.10f, 0.11f, 0.90f}});
  const std::vector<std::pair<int, int>> groups{{0, 2}, {2, 3}};
  const cv::Mat out = okvis::filterSuperPointMatches(d, groups, anyPair, 0.8, false);
  EXPECT_FLOAT_EQ(out.at<float>(0, 0), 0.10f);
  EXPECT_FLOAT_EQ(out.at<float>(0, 1), 0.11f); // the winner keeps all its descriptors
  EXPECT_EQ(out.at<float>(0, 2), kInf);
}

TEST(SuperPointMatchFilter, gateLimitsTheCompetitors) {
  // Candidate 1 is close in descriptor space but outside the gate, so it cannot make query 0
  // ambiguous, and it is removed.
  const cv::Mat d = matrix({{0.10f, 0.11f}});
  const cv::Mat out = okvis::filterSuperPointMatches(
      d, columns(2), [](int, int candidate) { return candidate == 0; }, 0.8, false);
  EXPECT_FLOAT_EQ(out.at<float>(0, 0), 0.10f);
  EXPECT_EQ(out.at<float>(0, 1), kInf);
}

TEST(SuperPointMatchFilter, disabledFiltersLeaveTheBestGatedMatchOnly) {
  const cv::Mat d = matrix({{0.10f, 0.11f}});
  const cv::Mat out = okvis::filterSuperPointMatches(d, columns(2), anyPair, 1.0, false);
  EXPECT_FLOAT_EQ(out.at<float>(0, 0), 0.10f);
  EXPECT_EQ(out.at<float>(0, 1), kInf);
}
