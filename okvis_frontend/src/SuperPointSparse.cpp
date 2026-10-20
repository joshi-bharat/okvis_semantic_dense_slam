/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#include <okvis/SuperPointSparse.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace okvis {

void selectSparseSuperPoint(const cv::Mat & keypoints, const cv::Mat & scores,
                            const cv::Mat & descriptors, float threshold, int radius,
                            int maxKeypoints, std::vector<cv::KeyPoint> & keypointsOut,
                            cv::Mat & descriptorsOut) {
  CV_Assert(keypoints.cols == 2 && (keypoints.depth() == CV_32F || keypoints.depth() == CV_32S));
  CV_Assert(scores.type() == CV_32F && int(scores.total()) == keypoints.rows);
  CV_Assert(descriptors.type() == CV_32F && descriptors.rows == keypoints.rows);
  CV_Assert(scores.isContinuous());
  keypointsOut.clear();

  cv::Mat points;
  keypoints.convertTo(points, CV_32F);
  const float * score = scores.ptr<float>();

  std::vector<int> order;
  order.reserve(size_t(points.rows));
  for (int i = 0; i < points.rows; ++i) {
    if (score[i] >= threshold) {
      order.push_back(i);
    }
  }
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    if (score[a] != score[b]) {
      return score[a] > score[b];
    }
    const float ya = points.at<float>(a, 1), yb = points.at<float>(b, 1);
    if (ya != yb) {
      return ya < yb;
    }
    return points.at<float>(a, 0) < points.at<float>(b, 0);
  });

  std::vector<int> kept;
  kept.reserve(std::min(order.size(), size_t(std::max(0, maxKeypoints))));
  for (const int i : order) {
    if (int(kept.size()) >= maxKeypoints) {
      break;
    }
    const float x = points.at<float>(i, 0);
    const float y = points.at<float>(i, 1);
    bool suppressed = false;
    for (const int k : kept) {
      if (std::abs(points.at<float>(k, 0) - x) <= float(radius)
          && std::abs(points.at<float>(k, 1) - y) <= float(radius)) {
        suppressed = true;
        break;
      }
    }
    if (radius > 0 && suppressed) {
      continue;
    }
    kept.push_back(i);
  }

  cv::Mat selected(int(kept.size()), descriptors.cols, CV_32F);
  for (int r = 0; r < int(kept.size()); ++r) {
    const int i = kept[size_t(r)];
    keypointsOut.emplace_back(cv::Point2f(points.at<float>(i, 0), points.at<float>(i, 1)), 1.0f,
                              -1.0f, score[i]);
    // Normalise first, convert second: fp16 must never see unnormalised magnitudes.
    const double norm = cv::norm(descriptors.row(i), cv::NORM_L2);
    descriptors.row(i).convertTo(selected.row(r), CV_32F, norm > 0.0 ? 1.0 / norm : 0.0);
  }
  selected.convertTo(descriptorsOut, CV_16F);
}

} // namespace okvis
