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
 * @file okvis_node_synchronous.cpp
 * @brief This file includes the ROS node implementation: synchronous (blocking) dataset processing.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <execinfo.h>
#include <unistd.h>
#include <Eigen/Core>
#include <fstream>
#include <thread>

#include <boost/filesystem.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#pragma GCC diagnostic pop

#include <okvis/TrajectoryOutput.hpp>
#include <okvis/ViParametersReader.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/DatasetReader.hpp>
#include <okvis/RpgDatasetReader.hpp>
#include <okvis/ros2/RosbagReader.hpp>
#include <okvis/TrajectoryOutput.hpp>
#include <okvis/ros2/Publisher.hpp>
#include <okvis/ThreadedPublisher.hpp>
#include <okvis/RerunVisualizer.hpp>

namespace {

/// Number of SIGINT/SIGTERM received. 1: stop gracefully, 2: exit immediately.
std::atomic_int g_interrupts{0};

void onInterrupt(int) {
  if (++g_interrupts >= 2) {
    static const char msg[] = "\nSecond interrupt -- exiting immediately.\n";
    const ssize_t ignored = write(STDERR_FILENO, msg, sizeof(msg) - 1); // async-signal-safe
    (void)ignored;
    std::_Exit(130);
  }
}

}  // namespace

/// \brief Main
/// \param argc argc.
/// \param argv argv.
int main(int argc, char **argv)
{

  // ros2 setup; own signal handling so that Ctrl+C stops reading instead of processing the rest
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, onInterrupt);
  std::signal(SIGTERM, onInterrupt);
  std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("okvis_node_synchronous");

  // publisher
  auto threadedOdometryPublisher = std::make_shared<okvis::ThreadedPublisher>(node);
  auto threadedImagePublisher = std::make_shared<okvis::ThreadedPublisher>(node);
  auto threadedPublisher = std::make_shared<okvis::ThreadedPublisher>(node);
  okvis::Publisher publisher(
    node,
    threadedOdometryPublisher,
    threadedImagePublisher,
    threadedPublisher);

  // logging
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  // Setting up paramaters
  okvis::Duration deltaT(0.0);
  bool rpg = false;
  bool rgb = false;
  bool rosbag = false;
  std::string configFilename("");
  std::string path("");
  std::string csvPath("");
  double startDelay = 0.0;
  okvis::RosbagReader::Topics bagTopics;

  node->declare_parameter("rpg", false);
  node->declare_parameter("rgb", false);
  node->declare_parameter("config_filename", "");
  node->declare_parameter("path", "");
  node->declare_parameter("csv_path", ""); // default: next to the dataset / bag
  node->declare_parameter("start_delay", 0.0); // [s] to skip at the beginning
  node->declare_parameter("imu_propagated_state_publishing_rate", 0.0);
  // ROS2 bag topics (remapping does not apply when reading a bag file)
  node->declare_parameter("imu_topic", bagTopics.imu);
  node->declare_parameter("cam_topics", std::vector<std::string>{});
  node->declare_parameter("depth_topics", std::vector<std::string>{});
  node->declare_parameter("ground_truth_topic", ""); // Odometry / PoseStamped / TransformStamped
  // visualisation
  node->declare_parameter("rerun", true); // log to Rerun (needs a build with -DUSE_RERUN=ON)
  node->declare_parameter("rerun_url", ""); // empty: spawn viewer; or rerun+http://host:9876/proxy, or file.rrd
  node->declare_parameter("show_images", true); // OpenCV windows with the feature-match overlays

  node->get_parameter("rpg", rpg);
  node->get_parameter("rgb", rgb);
  node->get_parameter("config_filename", configFilename);
  node->get_parameter("path", path);
  node->get_parameter("csv_path", csvPath);
  node->get_parameter("start_delay", startDelay);
  node->get_parameter("imu_topic", bagTopics.imu);
  node->get_parameter("cam_topics", bagTopics.cameras);
  node->get_parameter("depth_topics", bagTopics.depths);
  node->get_parameter("ground_truth_topic", bagTopics.groundTruth);
  bool useRerun = true;
  std::string rerunUrl;
  bool showImages = true;
  node->get_parameter("rerun", useRerun);
  node->get_parameter("rerun_url", rerunUrl);
  node->get_parameter("show_images", showImages);
  deltaT = okvis::Duration(startDelay);
  if (configFilename.compare("")==0){
    LOG(ERROR) << "ros parameter 'config_filename' not set";
    return EXIT_FAILURE;
  }
  if (path.compare("")==0){
    LOG(ERROR) << "ros parameter 'path' not set";
    return EXIT_FAILURE;
  }
  if (csvPath.empty()) {
    csvPath = path;
  }
  double imu_propagated_state_publishing_rate = 0.0;
  node->get_parameter("imu_propagated_state_publishing_rate", imu_propagated_state_publishing_rate);

  okvis::ViParametersReader viParametersReader(configFilename);
  okvis::ViParameters parameters;
  viParametersReader.getParameters(parameters);

  // Rerun visualisation (declared before reader and estimator, which call into it)
  okvis::RerunVisualizer rerun;
  if (useRerun && rerun.init("okvis2", rerunUrl)) {
    rerun.logCameras(parameters.nCameraSystem);
  }

  // dataset reader
  std::shared_ptr<okvis::DatasetReaderBase> dataset_reader;
  
  // check if ros2 bag
  std::ifstream f(path+"/metadata.yaml");
  if(f.good()) {
    rosbag = true;
  }
  if(rosbag) {
    for (size_t i = 0; i < parameters.nCameraSystem.numCameras(); ++i) {
      bagTopics.isColour.push_back(parameters.nCameraSystem.cameraType(i).isColour);
    }
    auto bagReader = std::make_shared<okvis::RosbagReader>(
      path, parameters.nCameraSystem.numCameras(),
      parameters.camera.sync_cameras, deltaT, bagTopics);
    if (!bagTopics.groundTruth.empty()) {
      bagReader->setGroundTruthCallback(
        [&rerun](const okvis::Time & t, const okvis::kinematics::Transformation & T_WS) {
          rerun.addGroundTruthPose(t, T_WS);
        });
    }
    dataset_reader = bagReader;
  } else if(rpg) {
    dataset_reader.reset(new okvis::RpgDatasetReader(
      path, deltaT, int(parameters.nCameraSystem.numCameras())));
  } else {
    dataset_reader.reset(new okvis::DatasetReader(
      path, int(parameters.nCameraSystem.numCameras()),
      parameters.camera.sync_cameras, deltaT));
  }

  // also check DBoW2 vocabulary
  boost::filesystem::path executable(argv[0]);
  std::string dBowVocDir = executable.remove_filename().string() + "/../../share/okvis/resources/";
  std::ifstream infile(dBowVocDir+"/small_voc.yml.gz");
  if(!infile.good()) {
    LOG(ERROR)<<"DBoW2 vocaublary " << dBowVocDir << "/small_voc.yml.gz not found.";
    return EXIT_FAILURE;
  }

  okvis::ThreadedSlam estimator(parameters, dBowVocDir);
  estimator.setBlocking(true);

  // write logs
  std::string mode = "slam";
  if(!parameters.estimator.do_loop_closures) {
    mode = "vio";
  }
  if(parameters.camera.online_calibration.do_extrinsics) {
    mode = mode+"-calib";
  }

  // setup publishing
  publisher.setCsvFile(csvPath + "/okvis2-" + mode + "-live_trajectory.csv", rpg);
  estimator.setFinalTrajectoryCsvFile(csvPath+"/okvis2-" + mode + "-final_trajectory.csv", rpg);
  estimator.setMapCsvFile(csvPath+"/okvis2-" + mode + "-final_map.csv");
  estimator.setOptimisedGraphCallback(
    [&publisher, &rerun](const okvis::State & state, const okvis::TrackingState & trackingState,
                         std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> updated,
                         std::shared_ptr<const okvis::MapPointVector> landmarks) {
      publisher.publishEstimatorUpdate(state, trackingState, updated, landmarks);
      rerun.logEstimatorUpdate(state, trackingState, updated, landmarks);
    });
  publisher.setBodyTransform(parameters.imu.T_BS);
  publisher.setOdometryPublishingRate(imu_propagated_state_publishing_rate);
  publisher.setupImageTopics(parameters.nCameraSystem);

  threadedOdometryPublisher->startThread();
  threadedImagePublisher->startThread();
  threadedPublisher->startThread();

  // connect reader to estimator
  dataset_reader->setImuCallback(
    std::bind(&okvis::ThreadedSlam::addImuMeasurement, &estimator,
              std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  dataset_reader->setImagesCallback(
    std::bind(&okvis::ThreadedSlam::addImages, &estimator, std::placeholders::_1,
              std::placeholders::_2, std::placeholders::_3));

  // start
  okvis::Time startTime = okvis::Time::now();
  if (!dataset_reader->startStreaming()) {
    LOG(ERROR) << "could not start reading " << path;
    return EXIT_FAILURE;
  }
  int progress = 0;
  while (rclcpp::ok() && g_interrupts == 0) {
    rclcpp::spin_some(node);

    estimator.processFrame();
    std::map<std::string, cv::Mat> images;
    estimator.display(images);
    publisher.publishImages(images);
    if (showImages && !images.empty()) {
      for (const auto & image : images) {
        cv::imshow(image.first, image.second);
      }
      cv::waitKey(1);
    }

    // check if done
    if(!dataset_reader->isStreaming()) {
      estimator.stopThreading();
      std::cout << "\rFinished!" << std::endl;
      if(parameters.estimator.do_final_ba) {
        LOG(INFO) << "final full BA...";
        cv::Mat topView;
        estimator.doFinalBa();
      }
      estimator.writeFinalTrajectoryCsv();
      if(parameters.estimator.do_final_ba) {
        estimator.saveMap();
      }
      LOG(INFO) <<"total processing time " << (okvis::Time::now() - startTime) << " s" << std::endl;
      break;
    }

    // display progress
    int newProgress = int(dataset_reader->completion()*100.0);
#ifndef DEACTIVATE_TIMERS
    if (newProgress>progress) {
      LOG(INFO) << okvis::timing::Timing::print();
    }
#endif
    if (newProgress>progress) {
      progress = newProgress;
      LOG(INFO) << "Progress: "
                << progress << "% "
                << std::flush;
    }
  }

  // interrupted (Ctrl+C): stop reading instead of letting the estimator drain the whole dataset.
  // The reader may be blocked handing a frame to the full estimator queue, so request the stop on a
  // separate thread while stopThreading() drains the queue.
  if (dataset_reader->isStreaming()) {
    LOG(WARNING) << "Interrupted -- stopping without final BA (Ctrl+C again to exit immediately)...";
    std::thread stopReader([&dataset_reader]() { dataset_reader->stopStreaming(); });
    estimator.stopThreading();
    stopReader.join();
    estimator.writeFinalTrajectoryCsv();
    LOG(INFO) << "Trajectory so far written to " << csvPath;
  }
  rclcpp::shutdown();
  return g_interrupts > 0 ? 130 : EXIT_SUCCESS;
}
