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
 * @file okvis2x_network_node_bag.cpp
 * @brief OKVIS2-X with volumetric submapping, reading a ROS2 bag sequentially -- the bag is read
 *        directly, there is no `ros2 bag play`. Depth comes from the stereo network (TensorRT
 *        engine or TorchScript model). Results go to the ROS graph and, optionally, to Rerun.
 */

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <vector>

#include <Eigen/Core>
#include <boost/filesystem.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#pragma GCC diagnostic pop

#include <okvis/Processor.hpp>
#include <okvis/RerunVisualizer.hpp>
#include <okvis/StereoDepthProcessorFactory.hpp>
#include <okvis/SubmappingInterface.hpp>
#include <okvis/ThreadedPublisher.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/ViParametersReader.hpp>
#include <okvis/timing/Timer.hpp>
#include <okvis/ros2/Publisher.hpp>
#include <okvis/ros2/RosbagReader.hpp>

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
int main(int argc, char **argv)
{
  // ros2 setup; own signal handling so that Ctrl+C stops reading instead of processing the rest
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, onInterrupt);
  std::signal(SIGTERM, onInterrupt);
  std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("okvis2x_network_node_bag");

  // logging
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  // parameters
  std::string configFilename;
  std::string seConfigFilename;
  std::string path;
  std::string csvPath;
  double startDelay = 0.0;
  okvis::RosbagReader::Topics bagTopics;

  node->declare_parameter("config_filename", "");
  node->declare_parameter("se_config_filename", "");
  node->declare_parameter("path", ""); // the ROS2 bag directory (the one with metadata.yaml)
  node->declare_parameter("csv_path", ""); // default: next to the bag
  node->declare_parameter("start_delay", 0.0); // [s] to skip at the beginning
  node->declare_parameter("imu_propagated_state_publishing_rate", 0.0);
  // ROS2 bag topics (remapping does not apply when reading a bag file)
  node->declare_parameter("imu_topic", bagTopics.imu);
  node->declare_parameter("cam_topics", std::vector<std::string>{});
  node->declare_parameter("depth_topics", std::vector<std::string>{});
  node->declare_parameter("ground_truth_topic", ""); // Odometry / PoseStamped / TransformStamped
  // mapping
  node->declare_parameter("mesh_cutoff_z", std::numeric_limits<double>::max());
  node->declare_parameter("save_submap_meshes", false);
  // visualisation
  node->declare_parameter("rerun", true); // also log to Rerun (needs a build with -DUSE_RERUN=ON)
  node->declare_parameter("rerun_url", ""); // empty: spawn viewer; or rerun+http://host:9876/proxy
  node->declare_parameter("show_images", true); // OpenCV windows with the overlays

  node->get_parameter("config_filename", configFilename);
  node->get_parameter("se_config_filename", seConfigFilename);
  node->get_parameter("path", path);
  node->get_parameter("csv_path", csvPath);
  node->get_parameter("start_delay", startDelay);
  node->get_parameter("imu_topic", bagTopics.imu);
  node->get_parameter("cam_topics", bagTopics.cameras);
  node->get_parameter("depth_topics", bagTopics.depths);
  node->get_parameter("ground_truth_topic", bagTopics.groundTruth);
  double meshCutoffZ = std::numeric_limits<double>::max();
  bool saveMeshes = false;
  bool useRerun = true;
  std::string rerunUrl;
  bool showImages = true;
  double imuPropagatedStatePublishingRate = 0.0;
  node->get_parameter("mesh_cutoff_z", meshCutoffZ);
  node->get_parameter("save_submap_meshes", saveMeshes);
  node->get_parameter("rerun", useRerun);
  node->get_parameter("rerun_url", rerunUrl);
  node->get_parameter("show_images", showImages);
  node->get_parameter("imu_propagated_state_publishing_rate", imuPropagatedStatePublishingRate);

  if (configFilename.empty()) {
    LOG(ERROR) << "ros parameter 'config_filename' not set";
    return EXIT_FAILURE;
  }
  if (seConfigFilename.empty()) {
    LOG(ERROR) << "ros parameter 'se_config_filename' not set";
    return EXIT_FAILURE;
  }
  if (path.empty()) {
    LOG(ERROR) << "ros parameter 'path' (the ROS2 bag directory) not set";
    return EXIT_FAILURE;
  }
  if (!std::ifstream(path + "/metadata.yaml").good()) {
    LOG(ERROR) << path << " is not a ROS2 bag directory (no metadata.yaml)";
    return EXIT_FAILURE;
  }
  if (csvPath.empty()) {
    csvPath = path;
  }
  const okvis::Duration deltaT(startDelay);

  // okvis parameters
  okvis::ViParametersReader viParametersReader(configFilename);
  okvis::ViParameters parameters;
  viParametersReader.getParameters(parameters);

  // supereight2 configuration
  se::SubMapConfig submapConfig(seConfigFilename);
  okvis::SupereightMapType::Config mapConfig;
  mapConfig.readYaml(seConfigFilename);
  okvis::SupereightMapType::DataConfigType dataConfig;
  dataConfig.readYaml(seConfigFilename);

  // publisher
  auto threadedOdometryPublisher = std::make_shared<okvis::ThreadedPublisher>(node);
  auto threadedImagePublisher = std::make_shared<okvis::ThreadedPublisher>(node);
  auto threadedPublisher = std::make_shared<okvis::ThreadedPublisher>(node);
  okvis::Publisher publisher(node, threadedOdometryPublisher, threadedImagePublisher,
                             threadedPublisher);
  publisher.setMeshCutoffZ(float(meshCutoffZ));

  // Rerun visualisation (declared before reader and processor, which call into it)
  okvis::RerunVisualizer rerun;
  if (useRerun && rerun.init("okvis2x", rerunUrl)) {
    rerun.logCameras(parameters.nCameraSystem);
  }

  // dataset reader: sequential ROS2 bag reading, no `ros2 bag play` needed
  for (size_t i = 0; i < parameters.nCameraSystem.numCameras(); ++i) {
    bagTopics.isColour.push_back(parameters.nCameraSystem.cameraType(i).isColour);
  }
  auto datasetReader = std::make_shared<okvis::RosbagReader>(
      path, parameters.nCameraSystem.numCameras(), parameters.camera.sync_cameras, deltaT,
      bagTopics);
  if (!bagTopics.groundTruth.empty()) {
    datasetReader->setGroundTruthCallback(
        [&rerun](const okvis::Time & t, const okvis::kinematics::Transformation & T_WS) {
          rerun.addGroundTruthPose(t, T_WS);
        });
  }

  // DBoW2 vocabulary / model directory: the installed resources, or next to the executable
  const boost::filesystem::path executable(argv[0]);
  const std::string binDir = boost::filesystem::path(executable).remove_filename().string();
  std::string dBowVocDir;
  for (const std::string & candidate : {binDir + "/../../share/okvis/resources",
                                        binDir + "/../share/okvis/resources", binDir}) {
    if (std::ifstream(candidate + "/small_voc.yml.gz").good()) {
      dBowVocDir = candidate;
      break;
    }
  }
  if (dBowVocDir.empty()) {
    LOG(ERROR) << "DBoW2 vocabulary small_voc.yml.gz not found relative to " << binDir;
    return EXIT_FAILURE;
  }

  // stereo depth network: TensorRT engine (USE_TENSORRT) or TorchScript depth-model.pt (USE_NN)
  okvis::DeepLearningProcessor* dlProcessor =
      okvis::createStereoDepthProcessor(parameters, dBowVocDir);

  okvis::Processor processor(parameters, dlProcessor, dBowVocDir, mapConfig, dataConfig,
                             submapConfig);
  processor.setBlocking(true); // synchronous: never drop a frame
  processor.setT_BS(parameters.imu.T_BS);

  // write logs
  std::string mode = parameters.estimator.do_loop_closures ? "slam" : "vio";
  if (parameters.camera.online_calibration.do_extrinsics) {
    mode = mode + "-calib";
  }
  publisher.setCsvFile(csvPath + "/okvis2x-" + mode + "-live_trajectory.csv", false);
  processor.slam_.setFinalTrajectoryCsvFile(
      csvPath + "/okvis2x-" + mode + "-final_trajectory.csv", false);

  // output publishing
  publisher.setBodyTransform(parameters.imu.T_BS);
  publisher.setOdometryPublishingRate(imuPropagatedStatePublishingRate);
  publisher.setupImageTopics(parameters.nCameraSystem);
  publisher.setupNetworkTopics("stereo");

  // state updates -> ROS and Rerun
  processor.setOptimizedGraphCallback(
      [&publisher, &rerun](
          const okvis::State & state, const okvis::TrackingState & trackingState,
          std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> updated,
          std::shared_ptr<const okvis::MapPointVector> landmarks) {
        publisher.publishEstimatorUpdate(state, trackingState, updated, landmarks);
        rerun.logEstimatorUpdate(state, trackingState, updated, landmarks);
      });
  processor.setAlignmentPublishCallback(
      [&publisher](const okvis::Time & timestamp, const okvis::kinematics::Transformation & T_WS,
                   const std::vector<Eigen::Vector3f,
                                     Eigen::aligned_allocator<Eigen::Vector3f>> & alignPointCloud,
                   bool isMapFrame) {
        publisher.publishAlignmentPointsAsCallback(timestamp, T_WS, alignPointCloud, isMapFrame);
      });

  // submap meshes -> ROS and Rerun. Marching cubes runs once per submap and is cached; only the
  // transform is re-read on every callback, so loop closures move the meshes without re-meshing.
  std::unordered_map<uint64_t, okvis::SupereightMapType::SurfaceMesh> meshCache;
  processor.setSubmapCallback(
      [&rerun, &publisher, &meshCache, meshCutoffZ, useRerun](
          okvis::AlignedUnorderedMap<uint64_t, okvis::kinematics::Transformation> submapPoseLookup,
          std::unordered_map<uint64_t, std::shared_ptr<okvis::SupereightMapType>> submapLookup,
          std::shared_ptr<okvis::ObjectMap> objectMap) {
        publisher.publishSubmapsAsCallback(submapPoseLookup, submapLookup, objectMap);
        if (!useRerun) {
          return;
        }
        constexpr size_t n = okvis::SupereightMapType::SurfaceMesh::value_type::num_vertexes;
        for (auto & submap : submapLookup) {
          if (meshCache.find(submap.first) == meshCache.end()) {
            meshCache[submap.first] = se::algorithms::marching_cube(submap.second->getOctree());
          }
          const auto & mesh = meshCache[submap.first];

          // map voxels -> world, as in Publisher::publishSubmapsAsCallback
          const Eigen::Matrix4f T_OW = submapPoseLookup[submap.first].T().cast<float>();
          Eigen::Matrix4f T_WM_scale = submap.second->getTWM().matrix();
          T_WM_scale.topLeftCorner<3, 3>() *= submap.second->getRes();
          const Eigen::Matrix4f T_OM = T_OW * T_WM_scale;

          std::vector<Eigen::Vector3f> vertices;
          std::vector<Eigen::Vector3f> colours;
          vertices.reserve(n * mesh.size());
          bool anyColour = false;
          for (size_t i = 0; i < mesh.size(); ++i) {
            Eigen::Vector3f v[n];
            float maxZ = -std::numeric_limits<float>::max();
            for (size_t j = 0; j < n; ++j) {
              v[j] = (T_OM * mesh[i].vertexes[j].homogeneous()).template head<3>();
              maxZ = std::max(maxZ, v[j].z());
            }
            if (maxZ > float(meshCutoffZ)) {
              continue;
            }
            for (size_t j = 0; j < n; ++j) {
              vertices.push_back(v[j]);
            }
#ifdef OKVIS_COLIDMAP
            if constexpr (okvis::SupereightMapType::SurfaceMesh::value_type::col_
                          == se::Colour::On) {
              anyColour = true;
              for (size_t j = 0; j < n; ++j) {
                if (mesh[i].colour.vertexes) {
                  const auto & c = mesh[i].colour.vertexes.value()[j];
                  colours.emplace_back(float(c.r) / 255.0f, float(c.g) / 255.0f,
                                       float(c.b) / 255.0f);
                } else {
                  // not yet coloured: neutral grey, the viewer still shades it by the normal
                  colours.emplace_back(0.7f, 0.7f, 0.7f);
                }
              }
            }
#endif
          }
          if (!anyColour) {
            // no colour in this map at all: let the visualiser shade the mesh by its normals
            colours.clear();
          }
          rerun.logSubmapMesh(submap.first, vertices, colours);
        }
      });

  threadedOdometryPublisher->startThread();
  threadedImagePublisher->startThread();
  threadedPublisher->startThread();

  // connect reader to processor: Processor::addImages feeds both the estimator and the network
  datasetReader->setImuCallback(
      std::bind(&okvis::Processor::addImuMeasurement, &processor, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3));
  datasetReader->setImagesCallback(
      [&processor](const okvis::Time & stamp, const std::map<size_t, cv::Mat> & images,
                   const std::map<size_t, cv::Mat> & depthImages) {
        if (images.empty()) {
          return false;
        }
        std::map<size_t, std::pair<okvis::Time, cv::Mat>> stampedImages;
        std::map<size_t, std::pair<okvis::Time, cv::Mat>> stampedDepthImages;
        for (const auto & image : images) {
          stampedImages[image.first] = std::make_pair(stamp, image.second);
        }
        for (const auto & depth : depthImages) {
          stampedDepthImages[depth.first] = std::make_pair(stamp, depth.second);
        }
        return processor.addImages(stampedImages, stampedDepthImages);
      });

  // start
  const okvis::Time startTime = okvis::Time::now();
  if (!datasetReader->startStreaming()) {
    LOG(ERROR) << "could not start reading " << path;
    return EXIT_FAILURE;
  }
  int progress = 0;
  while (rclcpp::ok() && g_interrupts == 0) {
    rclcpp::spin_some(node);

    processor.processFrame();
    std::map<std::string, cv::Mat> images;
    processor.display(images);
    publisher.publishImages(images);
    if (showImages && !images.empty()) {
      for (const auto & image : images) {
        cv::imshow(image.first, image.second);
      }
      cv::waitKey(1);
    }

    // check if done
    if (!datasetReader->isStreaming()) {
      std::cout << "\rFinished reading!" << std::endl;
      if (parameters.estimator.do_final_ba) {
        LOG(INFO) << "final full BA...";
        processor.slam_.doFinalBa();
      }
      processor.finish();
      processor.slam_.writeFinalTrajectoryCsv();
      if (saveMeshes) {
        LOG(INFO) << "saving the submap meshes...";
        processor.se_interface_.saveAllSubmapMeshes();
      }
      LOG(INFO) << "total processing time " << (okvis::Time::now() - startTime) << " s";
      break;
    }

    // display progress
    const int newProgress = int(datasetReader->completion() * 100.0);
    if (newProgress > progress) {
#ifndef DEACTIVATE_TIMERS
      LOG(INFO) << okvis::timing::Timing::print();
#endif
      progress = newProgress;
      LOG(INFO) << "Progress: " << progress << "% " << std::flush;
    }
  }

  // interrupted (Ctrl+C): stop reading instead of letting the processor drain the whole bag. The
  // reader may be blocked handing a frame to a full queue, so stop it on a separate thread.
  if (datasetReader->isStreaming()) {
    LOG(WARNING) << "Interrupted -- stopping without final BA (Ctrl+C again to exit immediately)...";
    std::thread stopReader([&datasetReader]() { datasetReader->stopStreaming(); });
    processor.finish();
    stopReader.join();
    processor.slam_.writeFinalTrajectoryCsv();
    LOG(INFO) << "Trajectory so far written to " << csvPath;
  }

  cv::destroyAllWindows();
  rclcpp::shutdown();
  return g_interrupts > 0 ? 130 : EXIT_SUCCESS;
}
