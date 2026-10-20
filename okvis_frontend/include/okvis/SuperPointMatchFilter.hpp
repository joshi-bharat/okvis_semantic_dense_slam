/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#ifndef INCLUDE_OKVIS_SUPERPOINTMATCHFILTER_HPP_
#define INCLUDE_OKVIS_SUPERPOINTMATCHFILTER_HPP_

#include <functional>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace okvis {

/**
 * @brief Lowe's ratio test and cross-check on a SuperPoint distance matrix.
 *
 * Rows are queries (current keypoints), columns are descriptors grouped into candidates (a map
 * landmark owns up to three stored descriptors; a keypoint of another frame owns one). A
 * query-candidate distance is the minimum over the candidate's descriptors; only pairs the gate
 * admits compete. Each query keeps its best candidate if
 *   - ratio test: L2(best) < ratio * L2(second best), i.e. d1 < ratio^2 * d2 for d = 1 - cos of
 *     unit descriptors (no second candidate: passes), and
 *   - cross-check: the query is also that candidate's best query.
 * All other entries become +inf, so the existing matching code, with its absolute threshold and
 * geometric checks, can only pick the surviving pair.
 *
 * @param distances CV_32F, queries x descriptors, d = 1 - cosine similarity.
 * @param groups Candidate c owns descriptor columns [groups[c].first, groups[c].second).
 * @param gate Whether query q may match candidate c at all (e.g. reprojection radius).
 * @param ratio Ratio test threshold on L2 distances; >= 1 disables it.
 * @param crossCheck Require mutual best matches.
 * @return The filtered copy of distances.
 */
cv::Mat filterSuperPointMatches(const cv::Mat & distances,
                                const std::vector<std::pair<int, int>> & groups,
                                const std::function<bool(int query, int candidate)> & gate,
                                double ratio, bool crossCheck);

} // namespace okvis

#endif // INCLUDE_OKVIS_SUPERPOINTMATCHFILTER_HPP_
