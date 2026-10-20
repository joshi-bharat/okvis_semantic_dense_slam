/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#include <okvis/SuperPointExtractor.hpp>

#include <algorithm>
#include <stdexcept>

#include <glog/logging.h>
#include <opencv2/imgproc.hpp>

#include <okvis/SuperPointDescriptors.hpp>
#include <okvis/SuperPointNms.hpp>
#include <okvis/TensorRtEngine.hpp>

namespace okvis {

namespace {
int64_t elementCount(const std::vector<int64_t> & shape) {
  int64_t count = 1;
  for (const int64_t dimension : shape) {
    count *= dimension;
  }
  return count;
}

/// @brief Engine output as CV_32F, whatever precision the engine was built with.
cv::Mat outputAsFloat(TensorRtEngine & engine, const std::string & name) {
  const cv::Mat raw = engine.getOutput(name);
  if (raw.depth() == CV_32F) {
    return raw;
  }
  cv::Mat asFloat;
  raw.convertTo(asFloat, CV_32F);
  return asFloat;
}
} // namespace

SuperPointExtractor::SuperPointExtractor(const SuperPointParameters & parameters,
                                         int maxKeypoints)
    : parameters_(parameters), maxKeypoints_(maxKeypoints) {
  engine_ = std::make_shared<TensorRtEngine>(parameters.engine);
  const auto inputs = engine_->inputs();
  if (inputs.size() != 1) {
    throw std::runtime_error("SuperPointExtractor: expected exactly one engine input, got "
                             + std::to_string(inputs.size()));
  }
  const auto & shape = inputs.front().shape;
  if (shape.size() < 2 || shape[shape.size() - 1] <= 0 || shape[shape.size() - 2] <= 0) {
    throw std::runtime_error("SuperPointExtractor: engine input needs a static HxW shape");
  }
  networkSize_ = cv::Size(int(shape[shape.size() - 1]), int(shape[shape.size() - 2]));
  const auto outputs = engine_->outputs();
  if (outputs.size() != 2) {
    throw std::runtime_error("SuperPointExtractor: expected two engine outputs (scores, "
                             "descriptors), got " + std::to_string(outputs.size()));
  }
  // Select by element count, not by order: the full-resolution score map (1 x H x W) is larger
  // than the coarse descriptor grid (D x H/8 x W/8) for D < 64, and the reference D is 256, so
  // the score map is the one whose element count equals the input's.
  const int64_t inputCount = int64_t(networkSize_.area());
  const bool firstIsScores = elementCount(outputs[0].shape) == inputCount;
  const std::string inputName = inputs.front().name;
  const std::string scoresName = outputs[firstIsScores ? 0 : 1].name;
  const std::string coarseName = outputs[firstIsScores ? 1 : 0].name;
  const auto coarseShape = outputs[firstIsScores ? 1 : 0].shape;
  if (elementCount(outputs[firstIsScores ? 0 : 1].shape) != inputCount || coarseShape.size() < 3) {
    throw std::runtime_error("SuperPointExtractor: expected a score map of the input's size and "
                             "a D x Hc x Wc descriptor grid");
  }
  const int dimension = int(coarseShape[coarseShape.size() - 3]);
  LOG(INFO) << "SuperPoint engine " << parameters.engine << ": input " << inputName << " "
            << networkSize_.width << "x" << networkSize_.height << ", scores " << scoresName
            << ", descriptors " << coarseName << " (D = " << dimension << ")";

  inference_ = [this, inputName, scoresName, coarseName, dimension](
                   const cv::Mat & input, cv::Mat & scores, cv::Mat & coarse) {
    engine_->setInput(inputName, input);
    engine_->infer();
    scores = outputAsFloat(*engine_, scoresName).reshape(1, networkSize_.height);
    coarse = outputAsFloat(*engine_, coarseName).reshape(1, dimension);
  };
}

SuperPointExtractor::SuperPointExtractor(const SuperPointParameters & parameters,
                                         int maxKeypoints, cv::Size networkSize,
                                         InferenceFunction inference)
    : parameters_(parameters), maxKeypoints_(maxKeypoints), networkSize_(networkSize),
      inference_(std::move(inference)) {}

SuperPointExtractor::~SuperPointExtractor() = default;

void SuperPointExtractor::extract(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints,
                                  cv::Mat & descriptors) {
  keypoints.clear();
  descriptors = cv::Mat();
  if (image.empty()) {
    return;
  }

  cv::Mat grey;
  if (image.channels() == 1) {
    grey = image;
  } else {
    cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
  }

  cv::Mat input;
  if (grey.size() != networkSize_) {
    cv::resize(grey, input, networkSize_, 0, 0, cv::INTER_LINEAR);
  } else {
    input = grey;
  }
  cv::Mat normalised;
  input.convertTo(normalised, CV_32F, 1.0 / 255.0);

  cv::Mat scores;
  cv::Mat coarse;
  inference_(normalised, scores, coarse);

  keypoints = superPointNms(scores, float(parameters_.threshold), parameters_.nms_radius,
                            maxKeypoints_);
  descriptors = sampleSuperPointDescriptors(coarse, keypoints, scores.size());

  // Back to image coordinates. Sampling happened in score-map space, so this must come after.
  const float scaleX = float(grey.cols) / float(scores.cols);
  const float scaleY = float(grey.rows) / float(scores.rows);
  for (cv::KeyPoint & keypoint : keypoints) {
    keypoint.pt.x = (keypoint.pt.x + 0.5f) * scaleX - 0.5f;
    keypoint.pt.y = (keypoint.pt.y + 0.5f) * scaleY - 0.5f;
    keypoint.size = 8.0f * std::max(scaleX, scaleY);
    keypoint.octave = 0;
  }
}

} // namespace okvis
