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
 * @file RosbagReader.hpp
 * @brief Header file for the DatasetReader class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_ROSBAGREADER_HPP_
#define INCLUDE_OKVIS_ROSBAGREADER_HPP_

#include <atomic>
#include <functional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <glog/logging.h>

#include <rclcpp/serialization.hpp>

#include <rosbag2_cpp/readers/sequential_reader.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/ViSensorBase.hpp>
#include <okvis/kinematics/Transformation.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/// @brief Topic names for RosbagReader, and how to decode images.
struct RosbagTopics {
  std::string imu = "/okvis/imu0"; ///< sensor_msgs/Imu topic.
  /// Image topic per camera. Empty entries (or a short vector) default to /okvis/cam<i>/image_raw.
  std::vector<std::string> cameras;
  /// Depth image topic per camera. Empty entries default to /okvis/depth<i>/image_raw.
  std::vector<std::string> depths;
  std::vector<bool> isColour; ///< Per camera: output rgb8 instead of mono8. Default: mono8.
  /// Optional ground-truth pose topic (nav_msgs/Odometry, geometry_msgs/PoseStamped or
  /// geometry_msgs/TransformStamped). Empty: none.
  std::string groundTruth;
};

/// @brief Reader class acting like a VI sensor, reading a ROS2 bag file sequentially.
///
/// Messages are consumed in bag order directly from disk (sqlite3 or mcap); nothing is replayed
/// or published, so combined with a blocking estimator no data is dropped.
/// Images may be sensor_msgs/Image in any cv_bridge-convertible encoding, or
/// sensor_msgs/CompressedImage; they are converted to mono8 (or rgb8 for colour cameras).
/// @warning Make sure to use this in combination with synchronous
/// processing, as there is no throttling of the reading process.
class RosbagReader : public DatasetReaderBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  /// @brief Topic names to read, and how to decode images.
  using Topics = RosbagTopics;

  /// @brief Disallow default construction.
  RosbagReader() = delete;

  /// @brief Construct pointing to dataset.
  /// @param path The absolute or relative path to the dataset.
  /// @param numCameras The total number of cameras.
  /// @param syncCameras Camera group to force synchronisation.
  /// @param deltaT Duration [s] to skip in the beginning.
  /// @param topics Topic names to read and image decoding options.
  RosbagReader(const std::string& path, size_t numCameras, const std::set<size_t> & syncCameras,
                const Duration & deltaT = Duration(0.0), const Topics & topics = Topics());

  /// @brief Destructor: stops streaming.
  virtual ~RosbagReader();

  /// @brief (Re-)setting the dataset path.
  /// @param path The absolute or relative path to the dataset.
  /// @return True, if the dateset folder structure could be created.
  virtual bool setDatasetPath(const std::string & path) final;

  /// @brief Setting skip duration in the beginning.
  /// @param deltaT Duration [s] to skip in the beginning.
  /// @return True on success.
  virtual bool setStartingDelay(const okvis::Duration & deltaT) final;

  /// @brief Starts reading the dataset.
  /// @return True, if successful
  virtual bool startStreaming() final;

  /// @brief Ground-truth pose callback: timestamp and pose in the ground-truth world frame.
  typedef std::function<void(const okvis::Time &, const okvis::kinematics::Transformation &)>
      GroundTruthCallback;

  /// @brief Set a callback for the ground-truth topic (Topics::groundTruth). Call before streaming.
  /// @param callback The callback.
  void setGroundTruthCallback(const GroundTruthCallback & callback) {
    groundTruthCallback_ = callback;
  }

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

  /// @brief Main processing loop.
  void processing();
  std::thread processingThread_; ///< Thread running processing loop.
  rosbag2_cpp::readers::SequentialReader reader_; ///< The ros2 bag reader.

  std::string path_; ///< Dataset path.

  std::atomic_bool streaming_; ///< Are we streaming?
  std::atomic_int counter_; ///< Number of images read yet.
  size_t numImages_ = 0; ///< Number of images to read.

  size_t numCameras_ = 0; ///< Total number of cameras.
  std::set<size_t> syncCameras_; ///< Camera group to force synchronisation.

  Topics topics_; ///< Resolved topic names (one per camera).
  std::vector<bool> camIsCompressed_; ///< Per camera: topic is sensor_msgs/CompressedImage.
  std::string groundTruthType_; ///< Message type of the ground-truth topic.
  GroundTruthCallback groundTruthCallback_; ///< Ground-truth callback.

  Duration deltaT_ = okvis::Duration(0.0); ///< Skip duration [s].

};

}

#endif // INCLUDE_OKVIS_ROSBAGREADER_HPP_
