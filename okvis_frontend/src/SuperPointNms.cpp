/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#include <okvis/SuperPointNms.hpp>

#include <algorithm>

namespace okvis {

std::vector<cv::KeyPoint> superPointNms(const cv::Mat & scores, float threshold, int radius,
                                        int maxKeypoints) {
  CV_Assert(scores.type() == CV_32F && scores.channels() == 1);
  std::vector<cv::KeyPoint> keypoints;
  if (radius < 0 || maxKeypoints <= 0) {
    return keypoints;
  }

  for (int v = radius; v < scores.rows - radius; ++v) {
    const float * row = scores.ptr<float>(v);
    for (int u = radius; u < scores.cols - radius; ++u) {
      const float score = row[u];
      if (score < threshold) {
        continue;
      }
      bool isMaximum = true;
      for (int dv = -radius; dv <= radius && isMaximum; ++dv) {
        const float * neighbourRow = scores.ptr<float>(v + dv);
        for (int du = -radius; du <= radius; ++du) {
          if (dv == 0 && du == 0) {
            continue;
          }
          const float neighbour = neighbourRow[u + du];
          // A strictly greater neighbour always wins. On a tie the earlier pixel (lower row,
          // then lower column) wins, so a plateau of equal scores collapses to one keypoint.
          const bool neighbourIsEarlier = dv < 0 || (dv == 0 && du < 0);
          if (neighbour > score || (neighbour == score && neighbourIsEarlier)) {
            isMaximum = false;
            break;
          }
        }
      }
      if (isMaximum) {
        keypoints.emplace_back(cv::Point2f(float(u), float(v)), 1.0f, -1.0f, score);
      }
    }
  }

  std::stable_sort(keypoints.begin(), keypoints.end(),
                   [](const cv::KeyPoint & a, const cv::KeyPoint & b) {
                     if (a.response != b.response) {
                       return a.response > b.response;
                     }
                     if (a.pt.y != b.pt.y) {
                       return a.pt.y < b.pt.y;
                     }
                     return a.pt.x < b.pt.x;
                   });
  if (int(keypoints.size()) > maxKeypoints) {
    keypoints.resize(size_t(maxKeypoints));
  }
  return keypoints;
}

} // namespace okvis
