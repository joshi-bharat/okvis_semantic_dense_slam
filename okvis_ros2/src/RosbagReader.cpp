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
 * @file RosbagReader.cpp
 * @brief Source file for the DatasetReader class.
 * @author Stefan Leutenegger
 */

#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_cpp/converter_interfaces/serialization_format_converter.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <image_transport/image_transport.hpp>
#if __has_include(<cv_bridge/cv_bridge.hpp>) // requires GCC >= 5
  #include <cv_bridge/cv_bridge.hpp>
#else
  #include <cv_bridge/cv_bridge.h> // ros2 changed to .hpp some point...
#endif
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <okvis/ViInterface.hpp>
#include <okvis/ros2/RosbagReader.hpp>

namespace okvis {

namespace {

/// @brief Decode a (possibly compressed) serialized image message to mono8 or rgb8.
cv::Mat decodeImage(rclcpp::SerializedMessage & serialized, bool compressed, bool colour,
                    okvis::Time & time) {
  if (compressed) {
    sensor_msgs::msg::CompressedImage msg;
    rclcpp::Serialization<sensor_msgs::msg::CompressedImage>().deserialize_message(&serialized, &msg);
    time = okvis::Time(msg.header.stamp.sec, msg.header.stamp.nanosec);
    cv::Mat image = cv::imdecode(msg.data, colour ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
    if (colour && !image.empty()) {
      cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    }
    return image;
  }
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  rclcpp::Serialization<sensor_msgs::msg::Image>().deserialize_message(&serialized, msg.get());
  time = okvis::Time(msg->header.stamp.sec, msg->header.stamp.nanosec);
  return cv_bridge::toCvCopy(msg, colour ? sensor_msgs::image_encodings::RGB8
                                         : sensor_msgs::image_encodings::MONO8)->image;
}

/// @brief Decode a serialized depth image message to CV_32FC1 [m].
cv::Mat decodeDepth(rclcpp::SerializedMessage & serialized, okvis::Time & time) {
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  rclcpp::Serialization<sensor_msgs::msg::Image>().deserialize_message(&serialized, msg.get());
  time = okvis::Time(msg->header.stamp.sec, msg->header.stamp.nanosec);
  if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1
      || msg->encoding == sensor_msgs::image_encodings::MONO16) {
    cv::Mat depth;
    cv_bridge::toCvShare(msg)->image.convertTo(depth, CV_32F, 0.001); // [mm] -> [m]
    return depth;
  }
  return cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1)->image;
}

}  // namespace

RosbagReader::RosbagReader(const std::string& path, size_t numCameras,
                           const std::set<size_t> &syncCameras, const Duration & deltaT,
                           const Topics & topics) :
    numCameras_(numCameras), syncCameras_(syncCameras), topics_(topics), deltaT_(deltaT) {
  streaming_ = false;
  setDatasetPath(path);
  counter_ = 0;

  // resolve default topic names
  topics_.cameras.resize(numCameras_);
  topics_.depths.resize(numCameras_);
  topics_.isColour.resize(numCameras_, false);
  for (size_t i = 0; i < numCameras_; ++i) {
    if (topics_.cameras[i].empty()) {
      topics_.cameras[i] = "/okvis/cam" + std::to_string(i) + "/image_raw";
    }
    if (topics_.depths[i].empty()) {
      topics_.depths[i] = "/okvis/depth" + std::to_string(i) + "/image_raw";
    }
  }
  camIsCompressed_.assign(numCameras_, false);
}

RosbagReader::~RosbagReader() {
  stopStreaming();
}

bool RosbagReader::setDatasetPath(const std::string & path) {
  path_ = path;
  return true;
}

bool RosbagReader::setStartingDelay(const Duration &deltaT)
{
  if(streaming_) {
    LOG(WARNING)<< "starting delay ignored, because streaming already started";
    return false;
  }
  deltaT_ = deltaT;
  return true;
}

bool RosbagReader::isStreaming()
{
  return streaming_;
}

double RosbagReader::completion() const {
  if(streaming_ && numImages_ > 0) {
    return double(counter_)/double(numImages_);
  }
  return 0.0;
}

bool RosbagReader::startStreaming() {
  OKVIS_ASSERT_TRUE(Exception, !imagesCallbacks_.empty(), "no add image callback registered")
  OKVIS_ASSERT_TRUE(Exception, !imuCallbacks_.empty(), "no add IMU callback registered")
  
  // set options (storage plugin, sqlite3 or mcap, is detected from the bag metadata)
  rosbag2_storage::StorageOptions storage_options{};
  storage_options.uri = path_;
  rosbag2_cpp::ConverterOptions converter_options{};
  converter_options.input_serialization_format = "cdr";
  converter_options.output_serialization_format = "cdr";

  // open bag
  reader_.open(storage_options, converter_options);
  
  // parse metadata
  const auto & metadata = reader_.get_metadata();
  size_t numImuMeasurements = 0;
  std::vector<size_t> numCamImages(numCameras_, 0);
  std::vector<size_t> numDepthImages(numCameras_, 0);
  for(const auto & info : metadata.topics_with_message_count) {
    const std::string & name = info.topic_metadata.name;
    if(name == topics_.imu) {
      numImuMeasurements = info.message_count;
    }
    if(!topics_.groundTruth.empty() && name == topics_.groundTruth) {
      groundTruthType_ = info.topic_metadata.type;
      LOG(INFO) << "No. ground-truth poses on " << name << " [" << groundTruthType_ << "]: "
                << info.message_count;
    }
    for(size_t i = 0; i < numCameras_; ++i) {
      if(name == topics_.cameras[i]) {
        numCamImages[i] = info.message_count;
        camIsCompressed_[i] = info.topic_metadata.type == "sensor_msgs/msg/CompressedImage";
      }
      if(name == topics_.depths[i]) {
        numDepthImages[i] = info.message_count;
      }
    }
  }
  numImages_ = numCamImages.empty() ? 0 : numCamImages[0];

  // print info
  LOG(INFO) << "Bag " << path_ << " (" << metadata.storage_identifier << ")";
  LOG(INFO) << "No. IMU measurements on " << topics_.imu << ": " << numImuMeasurements;
  for(size_t i = 0; i < numCameras_; ++i) {
    LOG(INFO) << "No. cam " << i << " images on " << topics_.cameras[i] << ": " << numCamImages[i]
              << (camIsCompressed_[i] ? " (compressed)" : "")
              << (topics_.isColour[i] ? " -> rgb8" : " -> mono8");
    if (numDepthImages[i] > 0) {
      LOG(INFO) << "No. cam " << i << " depth images on " << topics_.depths[i] << ": "
                << numDepthImages[i];
    }
  }
  bool missing = numImuMeasurements == 0;
  for(size_t i = 0; i < numCameras_; ++i) {
    missing |= numCamImages[i] == 0;
  }
  if (!topics_.groundTruth.empty()) {
    if (groundTruthType_.empty()) {
      LOG(WARNING) << "ground-truth topic " << topics_.groundTruth << " not in bag -- ignoring";
    } else if (groundTruthType_ != "nav_msgs/msg/Odometry"
               && groundTruthType_ != "geometry_msgs/msg/PoseStamped"
               && groundTruthType_ != "geometry_msgs/msg/TransformStamped") {
      LOG(WARNING) << "unsupported ground-truth type " << groundTruthType_ << " -- ignoring";
      groundTruthType_.clear();
    }
  }
  if (missing) {
    std::stringstream available;
    for(const auto & info : metadata.topics_with_message_count) {
      available << "\n  " << info.topic_metadata.name << " [" << info.topic_metadata.type << "] "
                << info.message_count;
    }
    LOG(ERROR) << "IMU or camera topic missing in bag. Available topics:" << available.str();
    return false;
  }

  counter_ = 0;
  streaming_ = true;
  processingThread_ = std::thread(&RosbagReader::processing, this);

  return true;
}

bool RosbagReader::stopStreaming() {
  // Stop the pipeline: the processing loop exits after its current (possibly blocking) callback
  streaming_ = false;
  if(processingThread_.joinable()) {
    processingThread_.join();
  }
  return true;
}

void  RosbagReader::processing() {
  okvis::Time start(0.0);
  okvis::Time t_imu(0.0);
  std::map<size_t, cv::Mat> images;
  std::map<size_t, cv::Mat> depthImages;
  std::map<size_t, cv::Mat> imagesSync;
  std::map<size_t, cv::Mat> depthImagesSync;
  std::map<size_t, okvis::Time> t_images;
  std::map<size_t, okvis::Time> t_depthImages;
  bool synced = false;

  while (reader_.has_next() && streaming_) {
    // serialize data
    auto serialized_message = reader_.read_next();
    rclcpp::SerializedMessage extracted_serialized_msg(*serialized_message->serialized_data);
    const std::string & topic = serialized_message->topic_name;
    
    // check if IMU
    if (topic == topics_.imu) {
      sensor_msgs::msg::Imu msg;
      rclcpp::Serialization<sensor_msgs::msg::Imu> serialization_info;
      serialization_info.deserialize_message(&extracted_serialized_msg, &msg);
      // construct measurement
      t_imu = okvis::Time(msg.header.stamp.sec, msg.header.stamp.nanosec);
      Eigen::Vector3d acc(msg.linear_acceleration.x, msg.linear_acceleration.y,
                      msg.linear_acceleration.z);
      Eigen::Vector3d gyr(msg.angular_velocity.x, msg.angular_velocity.y,
                      msg.angular_velocity.z); 
      
      // add it (keeping 1 s of IMU before the requested start, like the other readers)
      if (start == okvis::Time(0.0) || t_imu - start + okvis::Duration(1.0) > deltaT_) {
        for (auto &imuCallback : imuCallbacks_) {
          imuCallback(t_imu, acc, gyr);
        }
      }
      continue;
    }
    
    // check if ground truth
    if (!groundTruthType_.empty() && topic == topics_.groundTruth) {
      if (groundTruthCallback_) {
        okvis::Time t;
        Eigen::Vector3d r;
        Eigen::Quaterniond q;
        if (groundTruthType_ == "nav_msgs/msg/Odometry") {
          nav_msgs::msg::Odometry msg;
          rclcpp::Serialization<nav_msgs::msg::Odometry>().deserialize_message(
              &extracted_serialized_msg, &msg);
          t = okvis::Time(msg.header.stamp.sec, msg.header.stamp.nanosec);
          const auto & p = msg.pose.pose;
          r = Eigen::Vector3d(p.position.x, p.position.y, p.position.z);
          q = Eigen::Quaterniond(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
        } else if (groundTruthType_ == "geometry_msgs/msg/PoseStamped") {
          geometry_msgs::msg::PoseStamped msg;
          rclcpp::Serialization<geometry_msgs::msg::PoseStamped>().deserialize_message(
              &extracted_serialized_msg, &msg);
          t = okvis::Time(msg.header.stamp.sec, msg.header.stamp.nanosec);
          const auto & p = msg.pose;
          r = Eigen::Vector3d(p.position.x, p.position.y, p.position.z);
          q = Eigen::Quaterniond(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
        } else {
          geometry_msgs::msg::TransformStamped msg;
          rclcpp::Serialization<geometry_msgs::msg::TransformStamped>().deserialize_message(
              &extracted_serialized_msg, &msg);
          t = okvis::Time(msg.header.stamp.sec, msg.header.stamp.nanosec);
          const auto & tf = msg.transform;
          r = Eigen::Vector3d(tf.translation.x, tf.translation.y, tf.translation.z);
          q = Eigen::Quaterniond(tf.rotation.w, tf.rotation.x, tf.rotation.y, tf.rotation.z);
        }
        groundTruthCallback_(t, okvis::kinematics::Transformation(r, q.normalized()));
      }
      continue;
    }

    // check if image
    okvis::Time t_unsynced(0.0);
    for(size_t i = 0; i < numCameras_; ++i) {
      if (topic == topics_.cameras[i]) {
        okvis::Time time;
        cv::Mat image;
        try {
          image = decodeImage(extracted_serialized_msg, camIsCompressed_[i], topics_.isColour[i],
                              time);
        } catch (const std::exception & e) {
          LOG(WARNING) << "cam " << i << ": cannot decode image (" << e.what() << ") -- dropping";
          continue;
        }
        if (image.empty()) {
          LOG(WARNING) << "cam " << i << ": empty image at t=" << time << " -- dropping";
          continue;
        }
      
        if(syncCameras_.count(i)) {
          if(imagesSync.count(i)) {
            LOG(WARNING) << "image " << i << " at t=" << t_images.at(i)
                         << " without correspondence -- dropping";
          }
          imagesSync[i] = image;
        } else {
          images[i] = image;
          t_unsynced = time;
        }
        t_images[i] = time;
      }
      if (topic == topics_.depths[i]) {
        okvis::Time time;
        cv::Mat depthImage;
        try {
          depthImage = decodeDepth(extracted_serialized_msg, time);
        } catch (const std::exception & e) {
          LOG(WARNING) << "depth " << i << ": cannot decode image (" << e.what() << ") -- dropping";
          continue;
        }
        
        if(syncCameras_.count(i)) {
          if(depthImagesSync.count(i)) {
            LOG(WARNING) << "depth image " << i << " at t=" << t_depthImages.at(i)
                         << " without correspondence -- dropping";
          }
          depthImagesSync[i] = depthImage;
        } else {
          depthImages[i] = depthImage;
          t_unsynced = time;
        }
        t_depthImages[i] = time;
      }
    }
    
    // check if synced
    okvis::Time t_min(0.0);
    int i_min = 0;
    okvis::Time t_max(0.0);
    bool add = false;
    bool gotAllSyncImages = true;
    if(imagesSync.size() < syncCameras_.size()) {
      synced = false; // surely not synced yet, need to wait for all
    } else {
      // check timestamps
      bool first = true;
      for(int i : syncCameras_) {
        if(!imagesSync.count(i)) {
          synced = false;
          gotAllSyncImages = false;
          break;
        } else {
          if(t_images.at(i) < t_min || first) {
            t_min = t_images.at(i);
            i_min = i;
            first = false;
          }
          if(t_images.at(i) > t_max) {
            t_max = t_images.at(i);
          }
        }
      }
      if (gotAllSyncImages) {
        if (t_max - t_min > okvis::Duration(0.01)) { // 10 ms tolerance
          LOG(WARNING) << "image " << i_min << " at t=" << t_images.at(i_min)
                       << " without correspondence -- dropping";
          imagesSync.erase(i_min);
          t_images.erase(i_min);
          synced = false;
        } else {
          synced = true;
        }
      }
    }
    
    if(synced) {
      // move
      images.insert(imagesSync.begin(), imagesSync.end());
      depthImages.insert(depthImagesSync.begin(), depthImagesSync.end());
      imagesSync.clear();
      depthImagesSync.clear();
      add = true;
    } else {
      if(!images.empty() && !depthImages.empty()) {
        add = true;
      }
    }

    // add if required
    if(add) {
      okvis::Time t;
      if(synced) {
        t = t_min + okvis::Duration(0.5*(t_max-t_min).toSec());
      } else {
        t = t_unsynced;
      }
      if (start == okvis::Time(0.0)) {
        start = t;
      }

      // finally we are ready to call the image callback (unless still within the skip duration)
      if (t - start >= deltaT_) {
        for(auto & imagesCallback : imagesCallbacks_) {
          imagesCallback(t, images, depthImages);
        }
      }
      if(images.count(0) && !images.at(0).empty()) {
        ++counter_; // reference for counter is always image 0.
      }
      
      // clear
      synced = false;
      images.clear();
      depthImages.clear();
      t_images.clear();
      t_depthImages.clear();
    }
    
  }
  
  // done -- stop streaming
  streaming_ = false;

  return;
}

}
