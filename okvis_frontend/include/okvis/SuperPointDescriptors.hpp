/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#ifndef INCLUDE_OKVIS_SUPERPOINTDESCRIPTORS_HPP_
#define INCLUDE_OKVIS_SUPERPOINTDESCRIPTORS_HPP_

#include <vector>

#include <opencv2/core.hpp>

namespace okvis {

/**
 * @brief Sample SuperPoint's coarse descriptor grid at keypoint locations.
 *
 * The network emits descriptors on a grid coarser than the score map (1/8 for the reference
 * model). Each keypoint's descriptor is bilinearly interpolated from the four surrounding cells
 * and L2-normalised, so a dot product between two descriptors is their cosine similarity.
 *
 * @param coarse Descriptor grid, CV_32F, D rows by Hc*Wc columns (row-major cells).
 * @param keypoints Keypoints in score-map coordinates.
 * @param scoreMapSize Size of the score map the keypoints refer to.
 * @return CV_16F, keypoints.size() rows by D columns, each row of unit length.
 */
cv::Mat sampleSuperPointDescriptors(const cv::Mat & coarse,
                                    const std::vector<cv::KeyPoint> & keypoints,
                                    cv::Size scoreMapSize);

} // namespace okvis

#endif // INCLUDE_OKVIS_SUPERPOINTDESCRIPTORS_HPP_
