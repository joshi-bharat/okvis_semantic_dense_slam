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
 * @file TensorRtStereo2DepthProcessor.hpp
 * @brief Stereo depth prediction with a TensorRT engine (no libtorch).
 */

#ifndef OKVIS_TENSORRT_STEREO2DEPTH_PROCESSOR_HPP
#define OKVIS_TENSORRT_STEREO2DEPTH_PROCESSOR_HPP

#include <memory>
#include <set>
#include <string>

#include <opencv2/core/core.hpp>

#include <okvis/DeepLearningProcessor.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/TensorRtEngine.hpp>
#include <okvis/threadsafe/ThreadsafeQueue.hpp>

namespace okvis {

/**
 * @brief Drop-in replacement for Stereo2DepthProcessor that runs a TensorRT engine instead of a
 *        TorchScript model. Same inputs (rectified grey stereo pair), same outputs (depth [m] and
 *        1-sigma [m] images attached to the left camera measurement).
 *
 * Engine contract (defaults reproduce the TorchScript depth-model.pt contract):
 *  - Two image inputs, left and right: picked by name ("left"/"right"), otherwise in engine order.
 *    Layout [H,W], [C,H,W] or [1,C,H,W] with C = 1 or 3. With C = 3 and colour stereo cameras the
 *    images are fed as RGB (OKVIS images are BGR, so they are converted); grey images are
 *    replicated to the three channels.
 *    Element type uint8, float32 or float16; raw [0,255] pixel values are fed in either way, i.e.
 *    the network is expected to normalise internally (as S2M2's normalize_img does).
 *  - The engine must have a static input size (networks such as S2M2 need multiples of 32). The
 *    rectified images are resized to it, and the disparity is rescaled back to image pixels, so the
 *    calibrated focal length stays valid.
 *  - Disparity output: picked by name ("disp"), otherwise the first output. Its last two dimensions
 *    are H,W. Disparity is assumed to be in pixels of the output resolution and is rescaled to
 *    image pixels.
 *  - Disparity uncertainty, in this order: a second channel of the disparity output (as
 *    depth-model.pt), a separate output named "sigma"/"uncert"/"std", or an output named "conf"
 *    (as S2M2's output_conf). A confidence is a probability in (0,1), not a sigma, so it is mapped
 *    to a disparity 1-sigma; an "occ" output (S2M2's output_occ, a visibility probability) then
 *    additionally rejects occluded pixels. Without any of these, sigma is zero, i.e. the depth is
 *    trusted everywhere.
 *
 * All of this is configured in the tensorrt_parameters section of the OKVIS config file, see
 * okvis::TensorRtParameters.
 */
class TensorRtStereo2DepthProcessor : public DeepLearningProcessor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  /// @brief Construct and load the engine. Throws on failure.
  /// @param parameters OKVIS parameters (stereo camera indices, calibration, tensorrt_parameters).
  /// @param enginePath Serialised TensorRT engine.
  TensorRtStereo2DepthProcessor(okvis::ViParameters & parameters, const std::string & enginePath);

  /// @brief Stops the processing thread.
  virtual ~TensorRtStereo2DepthProcessor() override;

  /// @brief Display left/right images, disparity and sigma.
  virtual void display(std::map<std::string, cv::Mat> & images) override;

  /// @brief Add a set of new images (depthImages are ignored).
  virtual bool addImages(const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & images,
                         const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & depthImages)
      override final;

  /// @brief Check whether the processor is finished.
  bool finishedProcessing() override;

  /// @brief Run the engine on one stereo pair.
  /// @param left Left image (CV_8UC1 or CV_8UC3/BGR), already rectified.
  /// @param right Right image (CV_8UC1 or CV_8UC3/BGR), already rectified.
  /// @param[out] disparity Disparity [px], CV_32FC1, image size.
  /// @param[out] disparitySigma Disparity 1-sigma [px], CV_32FC1, image size (zeros if unavailable).
  void predictDisparity(const cv::Mat & left, const cv::Mat & right,
                        cv::Mat & disparity, cv::Mat & disparitySigma);

 protected:
  struct VisualizationData {
    okvis::Time timeStamp;
    cv::Mat leftImage;
    cv::Mat rightImage;
    cv::Mat depthImage;
    cv::Mat sigmaImage;
  };

  /// @brief Processing loop.
  void processing() override;

  /// @brief Predict depth for one synchronised frame set.
  void processStereoNetwork(std::map<size_t, std::vector<okvis::CameraMeasurement>> & frames);

  /// @brief Resolve tensor names and layouts from the engine and options.
  void configureEngine();

  /// @brief Convert a grey image to the engine input tensor.
  cv::Mat prepareInput(const std::string & name, const cv::Mat & image);

  std::unique_ptr<TensorRtEngine> engine_; ///< The TensorRT engine.
  TensorRtParameters options_; ///< Settings from the config file (tensorrt_parameters).

  // resolved engine layout
  std::string leftName_, rightName_, disparityName_, sigmaName_;
  std::string confidenceName_; ///< Confidence output in (0,1), if the engine has one.
  std::string occlusionName_; ///< Visibility (non-occlusion) probability output, if present.
  int inputChannels_ = 1; ///< Channels of the image inputs.
  bool colourInput_ = false; ///< Both stereo cameras are colour: feed RGB to a 3-channel engine.
  int netWidth_ = 0; ///< Network input width.
  int netHeight_ = 0; ///< Network input height.

  okvis::threadsafe::Queue<VisualizationData> visualisationsQueue_;

  double focalLength_ = 0.0; ///< Left focal length [px].
  double baseline_ = 0.0; ///< Stereo baseline [m].
  cv::Mat rectMapLeft0_, rectMapLeft1_, rectMapRight0_, rectMapRight1_;
  bool needRectify_ = false; ///< Rectify before the network.
  int imgWidth_ = 0; ///< Image width.
  int imgHeight_ = 0; ///< Image height.
  size_t idLeft_ = 0; ///< Left camera id.
  size_t idRight_ = 0; ///< Right camera id.
};

}  // namespace okvis

#endif  // OKVIS_TENSORRT_STEREO2DEPTH_PROCESSOR_HPP
