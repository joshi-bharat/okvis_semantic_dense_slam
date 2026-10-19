/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#ifndef INCLUDE_OKVIS_CUBLASMATCHER_HPP_
#define INCLUDE_OKVIS_CUBLASMATCHER_HPP_

#include <memory>

#include <opencv2/core.hpp>

namespace okvis {

/**
 * @brief Brute-force descriptor scoring on the GPU.
 *
 * Computes every query-train cosine similarity with a single half-precision GEMM. Descriptors
 * must already be L2-normalised, which is what makes the dot product a cosine similarity; this
 * class neither normalises nor thresholds. Geometry stays in the caller.
 *
 * Not thread safe: one instance per matching thread.
 */
class CublasMatcher {
public:
  /// @brief Creates the cuBLAS handle. Throws std::runtime_error if CUDA is unavailable.
  CublasMatcher();
  ~CublasMatcher();

  CublasMatcher(const CublasMatcher &) = delete;
  CublasMatcher & operator=(const CublasMatcher &) = delete;

  /**
   * @brief Similarity of every query descriptor against every train descriptor.
   * @param query CV_16F, N rows by D columns, L2-normalised.
   * @param train CV_16F, M rows by D columns, L2-normalised.
   * @return CV_32F, N by M; entry (i, j) is the cosine similarity of query i and train j.
   */
  cv::Mat score(const cv::Mat & query, const cv::Mat & train);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_; ///< cuBLAS handle and device buffers.
};

} // namespace okvis

#endif // INCLUDE_OKVIS_CUBLASMATCHER_HPP_
