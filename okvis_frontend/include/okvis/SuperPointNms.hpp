/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#ifndef INCLUDE_OKVIS_SUPERPOINTNMS_HPP_
#define INCLUDE_OKVIS_SUPERPOINTNMS_HPP_

#include <vector>

#include <opencv2/core.hpp>

namespace okvis {

/**
 * @brief Non-maximum suppression over a SuperPoint score map.
 *
 * The network emits a dense score per pixel; this reduces it to keypoints. A pixel survives when
 * it clears \p threshold, is the maximum within \p radius, and is at least \p radius from the
 * image border, where the coarse descriptor grid cannot be sampled without extrapolating. At
 * most \p maxKeypoints survive, the strongest ones.
 *
 * Equal scores break by row, then by column, so a cluster of identical scores yields one
 * keypoint rather than one per pixel, and rerunning on the same map gives the same result.
 *
 * @param scores Score map, CV_32F, single channel.
 * @param threshold Minimum score.
 * @param radius Suppression radius and border margin [pixels].
 * @param maxKeypoints Maximum number of keypoints returned.
 * @return Keypoints in score-map coordinates, response = score, sorted by descending score.
 */
std::vector<cv::KeyPoint> superPointNms(const cv::Mat & scores, float threshold, int radius,
                                        int maxKeypoints);

} // namespace okvis

#endif // INCLUDE_OKVIS_SUPERPOINTNMS_HPP_
