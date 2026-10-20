/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#ifndef INCLUDE_OKVIS_SUPERPOINTSPARSE_HPP_
#define INCLUDE_OKVIS_SUPERPOINTSPARSE_HPP_

#include <vector>

#include <opencv2/core.hpp>

namespace okvis {

/**
 * @brief Second, coarser NMS over the keypoints of an engine with built-in NMS and top-k.
 *
 * Engines exported with SuperPoint's own NMS (e.g. LightGlue-ONNX) return a fixed number of
 * keypoints, padded with weak ones when the image has fewer. A small in-engine radius keeps the
 * engine fast; this pass then enforces a wider spacing on the few candidates it returns.
 *
 * Keypoints below the threshold are dropped; the rest are visited strongest first (ties by row,
 * then column) and kept unless an already kept keypoint lies within radius in both x and y.
 *
 * @param keypoints N x 2 (x, y) in network pixels, CV_32F or CV_32S.
 * @param scores N scores, CV_32F, any shape.
 * @param descriptors N x D, CV_32F, one row per keypoint.
 * @param threshold Minimum score.
 * @param radius Suppression radius [pixels]; 0 disables suppression.
 * @param maxKeypoints Keypoint budget.
 * @param[out] keypointsOut Kept keypoints, strongest first, in network pixels.
 * @param[out] descriptorsOut CV_16F, one L2-normalised row per kept keypoint.
 */
void selectSparseSuperPoint(const cv::Mat & keypoints, const cv::Mat & scores,
                            const cv::Mat & descriptors, float threshold, int radius,
                            int maxKeypoints, std::vector<cv::KeyPoint> & keypointsOut,
                            cv::Mat & descriptorsOut);

} // namespace okvis

#endif // INCLUDE_OKVIS_SUPERPOINTSPARSE_HPP_
