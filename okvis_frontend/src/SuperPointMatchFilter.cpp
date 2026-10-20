/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#include <okvis/SuperPointMatchFilter.hpp>

#include <algorithm>
#include <limits>

namespace okvis {

cv::Mat filterSuperPointMatches(const cv::Mat & distances,
                                const std::vector<std::pair<int, int>> & groups,
                                const std::function<bool(int query, int candidate)> & gate,
                                double ratio, bool crossCheck) {
  CV_Assert(distances.type() == CV_32F);
  const float inf = std::numeric_limits<float>::infinity();
  const int numQueries = distances.rows;
  const int numCandidates = int(groups.size());

  std::vector<float> best(size_t(numQueries), inf);
  std::vector<float> second(size_t(numQueries), inf);
  std::vector<int> bestCandidate(size_t(numQueries), -1);
  std::vector<float> candidateBest(size_t(numCandidates), inf);
  std::vector<int> candidateBestQuery(size_t(numCandidates), -1);

  for (int q = 0; q < numQueries; ++q) {
    const float * row = distances.ptr<float>(q);
    for (int c = 0; c < numCandidates; ++c) {
      if (!gate(q, c)) {
        continue;
      }
      float d = inf;
      for (int col = groups[size_t(c)].first; col < groups[size_t(c)].second; ++col) {
        d = std::min(d, row[col]);
      }
      if (d < best[size_t(q)]) {
        second[size_t(q)] = best[size_t(q)];
        best[size_t(q)] = d;
        bestCandidate[size_t(q)] = c;
      } else if (d < second[size_t(q)]) {
        second[size_t(q)] = d;
      }
      if (d < candidateBest[size_t(c)]) {
        candidateBest[size_t(c)] = d;
        candidateBestQuery[size_t(c)] = q;
      }
    }
  }

  const double ratioSquared = ratio * ratio;
  cv::Mat filtered(distances.size(), CV_32F, cv::Scalar(inf));
  for (int q = 0; q < numQueries; ++q) {
    const int c = bestCandidate[size_t(q)];
    if (c < 0) {
      continue;
    }
    if (ratio < 1.0 && second[size_t(q)] < inf
        && !(double(best[size_t(q)]) < ratioSquared * double(second[size_t(q)]))) {
      continue;
    }
    if (crossCheck && candidateBestQuery[size_t(c)] != q) {
      continue;
    }
    const float * row = distances.ptr<float>(q);
    float * out = filtered.ptr<float>(q);
    for (int col = groups[size_t(c)].first; col < groups[size_t(c)].second; ++col) {
      out[col] = row[col];
    }
  }
  return filtered;
}

} // namespace okvis
