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
 * @file TumVieDatasetReader.hpp
 * @brief Header file for the TumVieDatasetReader class.
 */

#ifndef INCLUDE_OKVIS_TUMVIEDATASETREADER_HPP_
#define INCLUDE_OKVIS_TUMVIEDATASETREADER_HPP_

#include <atomic>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <glog/logging.h>

#include <okvis/assert_macros.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/ViSensorBase.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/**
 * @brief Reader for the TUM-VIE dataset (visual-inertial "*-vi_gt_data" sequences).
 *
 * Expected layout:
 * @code
 * <path>/imu_data.txt                           # t[us] gx gy gz [rad/s] ax ay az [m/s^2] temp
 * <path>/left_images/image_timestamps_left.txt  # one t[us] per line, line k <-> 0000k.jpg
 * <path>/left_images/%05d.jpg
 * <path>/right_images/image_timestamps_right.txt
 * <path>/right_images/%05d.jpg
 * @endcode
 * Timestamps are microseconds stored as floating point (with sub-microsecond fractions) and are
 * converted to nanoseconds. The IMU data is already calibrated by the dataset providers.
 * Camera 0 is left, camera 1 is right; both are hardware-triggered with identical timestamps.
 * If image_exposures_{left,right}.txt are present, the auto-exposure start-up burst (alternating
 * black and saturated frames) at the beginning of each sequence is skipped.
 *
 * @warning Use in combination with synchronous processing, as there is no throttling.
 */
class TumVieDatasetReader : public DatasetReaderBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  /// @brief Disallow default construction.
  TumVieDatasetReader() = delete;

  /// @brief Construct pointing to dataset.
  /// @param path The absolute or relative path to the sequence folder.
  /// @param deltaT Duration [s] to skip in the beginning.
  /// @param numCameras Number of cameras (1: left only, 2: stereo). If -1, use both.
  /// @param imuLookahead IMU measurements are streamed this far [s] ahead of each image. Must
  ///        cover the estimator's IMU overlap (0.02 s) plus any negative camera image_delay.
  TumVieDatasetReader(const std::string& path, const Duration & deltaT = Duration(0.0),
                      int numCameras = -1, const Duration & imuLookahead = Duration(0.05));

  /// @brief Destructor: stops streaming.
  virtual ~TumVieDatasetReader();

  /// @brief (Re-)setting the dataset path.
  /// @param path The absolute or relative path to the dataset.
  /// @return True, if the dateset folder structure could be created.
  virtual bool setDatasetPath(const std::string & path) final;

  /// @brief Setting skip duration in the beginning.
  /// deltaT Duration [s] to skip in the beginning.
  virtual bool setStartingDelay(const okvis::Duration & deltaT) final;

  /// @brief Starts reading the dataset.
  /// @return True, if successful
  virtual bool startStreaming() final;

  /// @brief Stops reading the dataset.
  /// @return True, if successful
  virtual bool stopStreaming() final;

  /// @brief Check if currently reading the dataset.
  /// @return True, if reading.
  virtual bool isStreaming() final;

  /// @brief Get the completion fraction read already.
  /// @return Fraction read already.
  virtual double completion() const final;

private:

  /// @brief A single parsed IMU line.
  struct ImuSample {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Time t; ///< Timestamp.
    Eigen::Vector3d acc; ///< Accelerometer [m/s^2].
    Eigen::Vector3d gyr; ///< Gyroscope [rad/s].
  };

  /// @brief Main processing loop.
  void processing();

  /// @brief Read the next non-comment IMU line.
  /// @param[out] sample The parsed sample.
  /// @return False at end of file.
  bool readImuSample(ImuSample & sample);

  /// @brief Parse a TUM-VIE timestamp file ([us] as float, '#' comments) into times.
  /// @param[in] filename The file.
  /// @param[out] times The timestamps.
  /// @return True on success.
  static bool readTimestamps(const std::string & filename, std::vector<Time> & times);

  /// @brief Convert floating-point microseconds to okvis::Time.
  /// @param us Microseconds.
  /// @return The time.
  static Time timeFromMicroseconds(double us);

  std::thread processingThread_; ///< Thread running processing loop.

  std::string path_; ///< Dataset path.

  std::atomic_bool streaming_; ///< Are we streaming?
  std::atomic_int counter_; ///< Number of frames read yet.
  size_t numImages_ = 0; ///< Number of frames to read.
  size_t firstImage_ = 0; ///< Index of first frame after the auto-exposure start-up burst.

  std::ifstream imuFile_; ///< IMU text file.

  Duration deltaT_ = okvis::Duration(0.0); ///< Skip duration [s].
  Duration imuLookahead_ = okvis::Duration(0.05); ///< IMU lookahead [s].

  std::vector<std::vector<Time>> imageTimes_; ///< Image timestamps per camera.
  std::vector<std::string> imageDirs_; ///< Image folder per camera.

  int numCameras_ = -1; ///< The number of cameras to consider (-1 = all).
};

}

#endif // INCLUDE_OKVIS_TUMVIEDATASETREADER_HPP_
