/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

/**
 * @file TensorRtStereo2DepthProcessor.cpp
 * @brief Stereo depth prediction with a TensorRT engine (no libtorch).
 */

#include <okvis/TensorRtStereo2DepthProcessor.hpp>

#include <algorithm>
#include <cmath>

#include <glog/logging.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <okvis/timing/Timer.hpp>
#include <okvis/cameras/PinholeCamera.hpp>

namespace okvis {

namespace {

bool contains(std::string haystack, const std::string & needle) {
  std::transform(haystack.begin(), haystack.end(), haystack.begin(), ::tolower);
  return haystack.find(needle) != std::string::npos;
}

/// Index of the H dimension (W follows) for a 2-4D image tensor; channel dim is before it (if any).
size_t heightDim(const std::vector<int64_t> & shape) {
  return shape.size() - 2;
}

// A confidence output (S2M2's output_conf) is a sigmoid probability that the pixel is correct, not
// a disparity uncertainty, so it has to be mapped to one: sigma = kSigmaAtFullConfidence / conf,
// and pixels the network itself would discard get a sigma large enough for the mapping to ignore
// them. The thresholds are S2M2's own (conf > 0.1 && occ > 0.5, see its visualize_stereo_results_2d).
constexpr float kSigmaAtFullConfidence = 0.5f; ///< Disparity 1-sigma [px] at confidence 1.
constexpr float kMinConfidence = 0.1f; ///< Below this the disparity is rejected.
constexpr float kMinVisibility = 0.5f; ///< Below this the pixel is considered occluded.
constexpr float kRejectedSigma = 100.0f; ///< Disparity 1-sigma [px] for rejected pixels.

}  // namespace

TensorRtStereo2DepthProcessor::TensorRtStereo2DepthProcessor(okvis::ViParameters & parameters,
                                                             const std::string & enginePath) {
  // stereo cameras and calibration (as Stereo2DepthProcessor, but colour cameras are allowed:
  // a 3-channel engine is fed RGB when the cameras are colour, replicated grey otherwise)
  if (parameters.camera.stereo_indices.size() != 2) {
    throw std::runtime_error("The stereo network needs exactly two stereo cameras");
  }
  colourInput_ = parameters.nCameraSystem.cameraType(parameters.camera.stereo_indices[0]).isColour
      && parameters.nCameraSystem.cameraType(parameters.camera.stereo_indices[1]).isColour;
  idLeft_ = parameters.camera.stereo_indices[0];
  idRight_ = parameters.camera.stereo_indices[1];

  Eigen::VectorXd intrinsics;
  kinematics::Transformation T_lr;
  needRectify_ = parameters.nCameraSystem.cameraType(idLeft_).depthType.needRectify;
  if (!needRectify_) {
    imgWidth_ = int(parameters.nCameraSystem.cameraGeometry(idLeft_)->imageWidth());
    imgHeight_ = int(parameters.nCameraSystem.cameraGeometry(idLeft_)->imageHeight());
    parameters.nCameraSystem.cameraGeometry(idLeft_)->getIntrinsics(intrinsics);
    T_lr = parameters.nCameraSystem.T_SC(idLeft_)->inverse()
        * (*parameters.nCameraSystem.T_SC(idRight_));
  } else {
    imgWidth_ = int(parameters.nCameraSystem.rectifyCameraGeometry(idLeft_)->imageWidth());
    imgHeight_ = int(parameters.nCameraSystem.rectifyCameraGeometry(idLeft_)->imageHeight());
    parameters.nCameraSystem.rectifyCameraGeometry(idLeft_)->getIntrinsics(intrinsics);
    T_lr = parameters.nCameraSystem.rectifyT_SC(idLeft_)->inverse()
        * (*parameters.nCameraSystem.rectifyT_SC(idRight_));
    parameters.nCameraSystem.rectifyCameraGeometry(idLeft_)
        ->getRectifyMap(rectMapLeft0_, rectMapLeft1_);
    parameters.nCameraSystem.rectifyCameraGeometry(idRight_)
        ->getRectifyMap(rectMapRight0_, rectMapRight1_);
  }
  focalLength_ = intrinsics(0);
  baseline_ = T_lr.r().norm();
  LOG(INFO) << "Stereo network (TensorRT) image width, height, focal length [pixel], and baseline [m] are "
            << imgWidth_ << ", " << imgHeight_ << ", " << focalLength_ << ", " << baseline_;

  // engine
  options_ = parameters.tensorrt;
  engine_.reset(new TensorRtEngine(enginePath));
  configureEngine();

  // warm up (allocations, kernel selection)
  cv::Mat zeros = cv::Mat::zeros(imgHeight_, imgWidth_, CV_8UC1);
  cv::Mat disparity, sigma;
  for (int i = 0; i < 3; ++i) {
    predictDisparity(zeros, zeros, disparity, sigma);
  }
  LOG(INFO) << "TensorRT stereo engine ready.";

  shutdown_ = false;
  isProcessing_ = false;
  processingThread_ = std::thread(&TensorRtStereo2DepthProcessor::processing, this);
}

TensorRtStereo2DepthProcessor::~TensorRtStereo2DepthProcessor() {
  cameraMeasurementsQueue_.Shutdown();
  visualisationsQueue_.Shutdown();
  shutdown_ = true;
  if (processingThread_.joinable()) {
    processingThread_.join();
  }
}

void TensorRtStereo2DepthProcessor::configureEngine() {
  const auto inputs = engine_->inputs();
  const auto outputs = engine_->outputs();
  if (inputs.size() < 2 || outputs.empty()) {
    throw std::runtime_error("stereo engine needs two inputs and at least one output:\n"
                             + engine_->describe());
  }

  // inputs
  leftName_.clear();
  rightName_.clear();
  for (const auto & input : inputs) {
    if (leftName_.empty() && contains(input.name, "left")) leftName_ = input.name;
    if (rightName_.empty() && contains(input.name, "right")) rightName_ = input.name;
  }
  if (leftName_.empty()) leftName_ = inputs[0].name;
  if (rightName_.empty()) rightName_ = inputs[leftName_ == inputs[0].name ? 1 : 0].name;

  // input layout: the engine has a static input size; images are resized to it
  std::vector<int64_t> shape = engine_->shape(leftName_);
  if (shape.size() < 2 || shape.size() > 4) {
    throw std::runtime_error("unsupported input rank for " + leftName_);
  }
  const size_t h = heightDim(shape);
  inputChannels_ = shape.size() >= 3 ? int(shape[h - 1]) : 1;
  if (inputChannels_ != 1 && inputChannels_ != 3) {
    throw std::runtime_error("stereo input " + leftName_ + " must have 1 or 3 channels");
  }
  if (shape[h] < 0 || shape[h + 1] < 0) {
    throw std::runtime_error("engine input " + leftName_ + " has a dynamic size "
                             + "-- build the engine with a fixed input size, e.g. trtexec "
                               "--shapes=" + leftName_ + ":1x" + std::to_string(inputChannels_)
                             + "x384x512");
  }
  netHeight_ = int(shape[h]);
  netWidth_ = int(shape[h + 1]);
  if (options_.network_width > 0 && options_.network_height > 0
      && (netWidth_ != options_.network_width || netHeight_ != options_.network_height)) {
    throw std::runtime_error("tensorrt_parameters network_width/network_height ("
                             + std::to_string(options_.network_width) + "x"
                             + std::to_string(options_.network_height)
                             + ") do not match the engine (" + std::to_string(netWidth_) + "x"
                             + std::to_string(netHeight_) + ")");
  }

  // outputs
  disparityName_.clear();
  sigmaName_.clear();
  for (const auto & output : outputs) {
    if (disparityName_.empty() && contains(output.name, "disp")) disparityName_ = output.name;
  }
  if (disparityName_.empty()) disparityName_ = outputs[0].name;
  confidenceName_.clear();
  occlusionName_.clear();
  for (const auto & output : outputs) {
    if (output.name == disparityName_) {
      continue;
    }
    if (contains(output.name, "sigma") || contains(output.name, "uncert")
        || contains(output.name, "std")) {
      sigmaName_ = output.name;
    } else if (contains(output.name, "conf")) {
      confidenceName_ = output.name;
    } else if (contains(output.name, "occ")) {
      occlusionName_ = output.name;
    }
  }

  // depth = f * B / disparity uses the image-resolution focal length, because predictDisparity()
  // rescales the network disparity back to image pixels (see toImagePixels there). For reference,
  // the focal length at the network's own resolution is:
  const double netFocalLength = focalLength_ * double(netWidth_) / double(imgWidth_);
  LOG(INFO) << "TensorRT stereo: focal length " << focalLength_ << " px at " << imgWidth_ << "x"
            << imgHeight_ << " (network sees " << netFocalLength << " px at " << netWidth_ << "x"
            << netHeight_ << "); disparity is rescaled to image pixels";
  LOG(INFO) << "TensorRT stereo: left=" << leftName_ << " right=" << rightName_
            << " (" << inputChannels_ << " ch, " << netWidth_ << "x" << netHeight_
            << ", images resized" << "), disparity=" << disparityName_
            << (sigmaName_.empty() ? "" : ", sigma=" + sigmaName_)
            << (confidenceName_.empty() ? "" : ", confidence=" + confidenceName_)
            << (occlusionName_.empty() ? "" : ", occlusion=" + occlusionName_);
  if (sigmaName_.empty() && confidenceName_.empty() && engine_->shape(disparityName_).size() < 3) {
    LOG(WARNING) << "TensorRT stereo: the engine has no uncertainty output -- the depth will be "
                    "reported with zero sigma, i.e. fully trusted everywhere.";
  }
}

cv::Mat TensorRtStereo2DepthProcessor::prepareInput(const std::string & name,
                                                    const cv::Mat & image) {
  if (image.cols != imgWidth_ || image.rows != imgHeight_) {
    throw std::runtime_error("stereo image is " + std::to_string(image.cols) + "x"
                             + std::to_string(image.rows) + ", expected (rectified) "
                             + std::to_string(imgWidth_) + "x" + std::to_string(imgHeight_));
  }
  // S2M2 and friends are trained on RGB; OKVIS images are BGR. Keep the colour when both the
  // cameras and the engine have three channels, otherwise fall back to (replicated) grey.
  const bool useColour = colourInput_ && inputChannels_ == 3 && image.channels() == 3;
  cv::Mat converted;
  if (useColour) {
    cv::cvtColor(image, converted, cv::COLOR_BGR2RGB);
  } else if (image.channels() == 3) {
    cv::cvtColor(image, converted, cv::COLOR_BGR2GRAY);
  } else {
    converted = image;
  }
  cv::Mat sized;
  cv::resize(converted, sized, cv::Size(netWidth_, netHeight_), 0, 0, cv::INTER_LINEAR);

  const TensorRtEngine::DataType type = [&]() {
    for (const auto & input : engine_->inputs()) {
      if (input.name == name) return input.dataType;
    }
    return TensorRtEngine::DataType::Other;
  }();
  const int depth = TensorRtEngine::cvDepth(type);
  if (depth != CV_8U && depth != CV_32F && depth != CV_16F) {
    throw std::runtime_error("unsupported input type for " + name);
  }

  // planar layout: C x H x W
  std::vector<cv::Mat> planes;
  if (sized.channels() == inputChannels_ && inputChannels_ > 1) {
    cv::split(sized, planes);
  } else {
    planes.assign(size_t(inputChannels_), sized); // grey, replicated if the engine wants 3 channels
  }
  cv::Mat tensor(1, inputChannels_ * netHeight_ * netWidth_, depth);
  for (int c = 0; c < inputChannels_; ++c) {
    cv::Mat plane(netHeight_, netWidth_, depth, tensor.ptr() + c * netHeight_ * netWidth_ * tensor.elemSize());
    if (depth == CV_8U) {
      planes[size_t(c)].copyTo(plane);
    } else {
      cv::Mat f;
      planes[size_t(c)].convertTo(f, CV_32F);
      if (depth == CV_16F) {
        f.convertTo(plane, CV_16F);
      } else {
        f.copyTo(plane);
      }
    }
  }
  return tensor;
}

void TensorRtStereo2DepthProcessor::predictDisparity(const cv::Mat & left, const cv::Mat & right,
                                                     cv::Mat & disparity,
                                                     cv::Mat & disparitySigma) {
  engine_->setInput(leftName_, prepareInput(leftName_, left));
  engine_->setInput(rightName_, prepareInput(rightName_, right));
  engine_->infer();

  // output -> CV_32F planes of H_out x W_out
  auto toPlanes = [this](const std::string & name) {
    const std::vector<int64_t> s = engine_->shape(name);
    if (s.size() < 2) {
      throw std::runtime_error("output " + name + " must have at least 2 dimensions");
    }
    const int h = int(s[s.size() - 2]);
    const int w = int(s[s.size() - 1]);
    cv::Mat raw = engine_->getOutput(name);
    cv::Mat values;
    raw.convertTo(values, CV_32F);
    const int channels = int(values.total()) / (h * w);
    std::vector<cv::Mat> planes;
    for (int c = 0; c < channels; ++c) {
      planes.push_back(cv::Mat(h, w, CV_32F, values.ptr<float>() + c * h * w).clone());
    }
    return planes;
  };

  std::vector<cv::Mat> disparityPlanes = toPlanes(disparityName_);
  cv::Mat disp = disparityPlanes[0];
  cv::Mat sigma;
  if (!sigmaName_.empty()) {
    sigma = toPlanes(sigmaName_)[0];
  } else if (disparityPlanes.size() >= 2) {
    sigma = disparityPlanes[1];
  } else if (!confidenceName_.empty()) {
    const cv::Mat confidence = toPlanes(confidenceName_)[0];
    cv::Mat clamped;
    cv::max(confidence, kMinConfidence, clamped);
    sigma = cv::Mat(kSigmaAtFullConfidence / clamped);
    cv::Mat rejected = confidence < kMinConfidence;
    if (!occlusionName_.empty()) {
      const cv::Mat occluded = toPlanes(occlusionName_)[0] < kMinVisibility;
      cv::bitwise_or(rejected, occluded, rejected);
    }
    sigma.setTo(kRejectedSigma, rejected);
  }

  // back to image resolution: disparity is in output pixels, so it scales with the width ratio
  const double toImagePixels = double(imgWidth_) / double(disp.cols);
  auto restore = [&](const cv::Mat & m) {
    cv::Mat out;
    cv::resize(m, out, cv::Size(imgWidth_, imgHeight_), 0, 0, cv::INTER_LINEAR);
    return cv::Mat(out * toImagePixels);
  };
  disparity = restore(disp);
  disparitySigma =
      sigma.empty() ? cv::Mat::zeros(imgHeight_, imgWidth_, CV_32F) : restore(sigma);
}

void TensorRtStereo2DepthProcessor::display(std::map<std::string, cv::Mat> & images) {
  if (visualisationsQueue_.Empty()) {
    return;
  }
  VisualizationData visData;
  if (visualisationsQueue_.PopNonBlocking(&visData)) {
    images["leftImage"] = visData.leftImage;
    images["rightImage"] = visData.rightImage;
    images["stereoDepth"] = visData.depthImage;
    if (!visData.sigmaImage.empty()) {
      images["stereoSigma"] = visData.sigmaImage;
    }
  }
}

bool TensorRtStereo2DepthProcessor::addImages(
    const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & images,
    const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & /*depthImages*/) {
  std::map<size_t, std::vector<okvis::CameraMeasurement>> cameraMeasurements;
  for (const auto & it : images) {
    okvis::CameraMeasurement camMeasurement;
    camMeasurement.measurement.image = it.second.second;
    camMeasurement.timeStamp = it.second.first;
    camMeasurement.sensorId = it.first;
    cameraMeasurements[it.first] = {camMeasurement};
  }
  if (blocking_) {
    cameraMeasurementsQueue_.PushBlockingIfFull(cameraMeasurements, 1);
    return true;
  }
  const int queueSize = 10;
  if (cameraMeasurementsQueue_.PushNonBlockingDroppingIfFull(cameraMeasurements, queueSize)) {
    DLOG(WARNING) << "stereo depth frame drop";
    return false;
  }
  return true;
}

void TensorRtStereo2DepthProcessor::processStereoNetwork(
    std::map<size_t, std::vector<okvis::CameraMeasurement>> & frames) {
  okvis::TimerSwitchable tAll("DNN 1 Stereo depth (TensorRT)");
  isProcessing_ = true;

  auto & frame0 = frames.at(idLeft_).front();
  auto & frame1 = frames.at(idRight_).front();
  if (needRectify_) {
    cv::Mat rectLeft, rectRight;
    cv::remap(frame0.measurement.image, rectLeft, rectMapLeft0_, rectMapLeft1_, cv::INTER_LINEAR);
    cv::remap(frame1.measurement.image, rectRight, rectMapRight0_, rectMapRight1_, cv::INTER_LINEAR);
    frame0.measurement.image = rectLeft;
    frame1.measurement.image = rectRight;
  }

  okvis::TimerSwitchable tInfer("DNN 1.2 Actual model inference (TensorRT)");
  cv::Mat disparity, disparitySigma;
  predictDisparity(frame0.measurement.image, frame1.measurement.image, disparity, disparitySigma);
  tInfer.stop();

  // same conversion as Stereo2DepthProcessor
  cv::Mat depth = float(focalLength_ * baseline_) / disparity;
  cv::Mat sigma = 2.0f * depth.mul(disparitySigma) / disparity;
  cv::patchNaNs(sigma, 100.0);
  cv::Mat infinite = (sigma == std::numeric_limits<float>::infinity())
      | (sigma == -std::numeric_limits<float>::infinity());
  sigma.setTo(100.0f, infinite);
  frame0.measurement.depthImage = depth;
  frame0.measurement.sigmaImage = sigma;

  if (imageCallback_) {
    imageCallback_(frames);
  }

  // visualisation
  cv::Mat visDisparity, visSigma;
  cv::Mat(cv::min(cv::max(disparity / 60.0f, 0.0f), 1.0f) * 255.0f).convertTo(visDisparity, CV_8U);
  cv::applyColorMap(visDisparity, visDisparity, cv::COLORMAP_INFERNO);
  cv::Mat(cv::min(cv::max(disparitySigma / 10.0f, 0.0f), 1.0f) * 255.0f).convertTo(visSigma, CV_8U);
  cv::applyColorMap(visSigma, visSigma, cv::COLORMAP_JET);
  VisualizationData visData;
  visData.timeStamp = frame0.timeStamp;
  visData.leftImage = frame0.measurement.image;
  visData.rightImage = frame1.measurement.image;
  visData.depthImage = visDisparity;
  visData.sigmaImage = visSigma;
  visualisationsQueue_.PushNonBlockingDroppingIfFull(visData, 1);

  isProcessing_ = false;
  tAll.stop();
}

void TensorRtStereo2DepthProcessor::processing() {
  // blocking pop in both modes (blocking_ only affects addImages); returns false on shutdown
  std::map<size_t, std::vector<okvis::CameraMeasurement>> frame;
  while (!shutdown_ && cameraMeasurementsQueue_.PopBlocking(&frame)) {
    processStereoNetwork(frame);
  }
}

bool TensorRtStereo2DepthProcessor::finishedProcessing() {
  return cameraMeasurementsQueue_.Size() == 0 && !isProcessing_;
}

}  // namespace okvis
