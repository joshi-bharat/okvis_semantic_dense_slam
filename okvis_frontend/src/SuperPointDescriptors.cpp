/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#include <okvis/SuperPointDescriptors.hpp>

#include <algorithm>
#include <cmath>

namespace okvis {

cv::Mat sampleSuperPointDescriptors(const cv::Mat & coarse,
                                    const std::vector<cv::KeyPoint> & keypoints,
                                    cv::Size scoreMapSize) {
  CV_Assert(coarse.type() == CV_32F);
  const int dimension = coarse.rows;
  const int cells = coarse.cols;
  if (keypoints.empty()) {
    return cv::Mat(0, dimension, CV_16F);
  }

  // Infer the grid shape from the score map's aspect: the network downsamples both axes equally.
  const int gridWidth = int(std::lround(std::sqrt(double(cells) * double(scoreMapSize.width)
                                                  / double(scoreMapSize.height))));
  CV_Assert(gridWidth > 0 && cells % gridWidth == 0);
  const int gridHeight = cells / gridWidth;

  const double scaleU = double(gridWidth) / double(scoreMapSize.width);
  const double scaleV = double(gridHeight) / double(scoreMapSize.height);

  cv::Mat descriptors(int(keypoints.size()), dimension, CV_32F);
  for (size_t k = 0; k < keypoints.size(); ++k) {
    // Map to grid coordinates, where cell centres sit at integer indices.
    const double u = (double(keypoints[k].pt.x) + 0.5) * scaleU - 0.5;
    const double v = (double(keypoints[k].pt.y) + 0.5) * scaleV - 0.5;
    const int u0 = std::max(0, std::min(gridWidth - 1, int(std::floor(u))));
    const int v0 = std::max(0, std::min(gridHeight - 1, int(std::floor(v))));
    const int u1 = std::min(gridWidth - 1, u0 + 1);
    const int v1 = std::min(gridHeight - 1, v0 + 1);
    const float du = float(std::clamp(u - double(u0), 0.0, 1.0));
    const float dv = float(std::clamp(v - double(v0), 0.0, 1.0));
    const float w00 = (1.0f - du) * (1.0f - dv);
    const float w01 = du * (1.0f - dv);
    const float w10 = (1.0f - du) * dv;
    const float w11 = du * dv;

    float * out = descriptors.ptr<float>(int(k));
    double squaredNorm = 0.0;
    for (int d = 0; d < dimension; ++d) {
      const float * plane = coarse.ptr<float>(d);
      const float value = w00 * plane[v0 * gridWidth + u0] + w01 * plane[v0 * gridWidth + u1]
                          + w10 * plane[v1 * gridWidth + u0] + w11 * plane[v1 * gridWidth + u1];
      out[d] = value;
      squaredNorm += double(value) * double(value);
    }
    const float inverseNorm = squaredNorm > 0.0 ? float(1.0 / std::sqrt(squaredNorm)) : 0.0f;
    for (int d = 0; d < dimension; ++d) {
      out[d] *= inverseNorm;
    }
  }

  // Normalise first, convert second: fp16 must never see unnormalised magnitudes.
  cv::Mat half;
  descriptors.convertTo(half, CV_16F);
  return half;
}

} // namespace okvis
