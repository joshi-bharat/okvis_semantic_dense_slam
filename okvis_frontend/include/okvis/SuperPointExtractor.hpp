/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#ifndef INCLUDE_OKVIS_SUPERPOINTEXTRACTOR_HPP_
#define INCLUDE_OKVIS_SUPERPOINTEXTRACTOR_HPP_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <okvis/Parameters.hpp>

namespace okvis {

class TensorRtEngine;

/**
 * @brief SuperPoint keypoints and descriptors for one camera.
 *
 * One forward pass per image yields a score map and a coarse descriptor grid; keypoints come
 * from non-maximum suppression in C++, descriptors from bilinear sampling of that grid. Keypoint
 * coordinates are returned in the original image resolution, whatever the network input size.
 *
 * Not thread safe: one instance per camera, as with the BRISK extractors.
 */
class SuperPointExtractor {
public:
  /// @brief Runs the network: image in, score map and coarse descriptor grid out.
  using InferenceFunction =
      std::function<void(const cv::Mat & input, cv::Mat & scores, cv::Mat & coarse)>;

  /**
   * @brief Construct from a serialised TensorRT engine.
   * @param parameters Threshold, NMS radius, engine path.
   * @param maxKeypoints Keypoint budget, i.e. frontend_parameters/max_num_keypoints.
   */
  SuperPointExtractor(const SuperPointParameters & parameters, int maxKeypoints);

  /// @brief Construct with an injected inference function, for testing without a GPU.
  SuperPointExtractor(const SuperPointParameters & parameters, int maxKeypoints,
                      cv::Size networkSize, InferenceFunction inference);

  ~SuperPointExtractor();

  /**
   * @brief Detect and describe.
   * @param[in] image Greyscale or colour image, any size.
   * @param[out] keypoints Keypoints in image coordinates, strongest first.
   * @param[out] descriptors CV_16F, one L2-normalised row per keypoint.
   */
  void extract(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints,
               cv::Mat & descriptors);

private:
  SuperPointParameters parameters_; ///< Threshold, radius, engine path.
  int maxKeypoints_; ///< Keypoint budget.
  cv::Size networkSize_; ///< Engine input size [pixels].
  InferenceFunction inference_; ///< Network call.
  std::shared_ptr<TensorRtEngine> engine_; ///< Null when an inference function was injected.
};

} // namespace okvis

#endif // INCLUDE_OKVIS_SUPERPOINTEXTRACTOR_HPP_
