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
 * @file RerunVisualizer.hpp
 * @brief Header file for the RerunVisualizer class.
 */

#ifndef INCLUDE_OKVIS_RERUNVISUALIZER_HPP_
#define INCLUDE_OKVIS_RERUNVISUALIZER_HPP_

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <okvis/Time.hpp>
#include <okvis/ViInterface.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/cameras/NCameraSystem.hpp>

#ifdef OKVIS_USE_RERUN
namespace rerun {
class RecordingStream;
}
#endif

/// \brief okvis Main namespace of this package.
namespace okvis {

/**
 * @brief Logs the OKVIS estimate to a Rerun viewer (https://rerun.io).
 *
 * Entities (world is gravity-aligned, z up):
 *   world/body            current pose T_WS
 *   world/body/cam<i>     camera frustum (T_SC + pinhole intrinsics)
 *   world/trajectory      optimised trajectory (updated on loop closure)
 *   world/keyframes       keyframe positions
 *   world/landmarks       sparse landmarks
 *   world/ground_truth    ground-truth path, if provided (aligned to the estimate, see below)
 * Data is stamped on the "sensor_time" timeline. Images are not logged.
 *
 * Ground truth lives in its own world frame. Both frames are gravity-aligned, so the ground truth
 * is aligned with a yaw + translation (4 DoF) transform computed at the first estimated state.
 * The ground-truth body frame is assumed to be the IMU frame S.
 *
 * All methods are no-ops when OKVIS is built without USE_RERUN, so callers need no ifdefs.
 * Methods are thread-safe; the estimator callback and display loop may run on different threads.
 */
class RerunVisualizer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// @brief Constructor (does not connect yet).
  RerunVisualizer();

  /// @brief Destructor: flushes pending data.
  ~RerunVisualizer();

  /// @brief Whether this build supports Rerun (built with USE_RERUN).
  static bool available();

  /// @brief Start logging.
  /// @param applicationId Rerun application ID shown in the viewer.
  /// @param connectUrl If empty, spawn a local viewer (needs `rerun` in PATH, e.g.
  ///        `pip install rerun-sdk==<SDK version>`); otherwise connect to a running viewer,
  ///        e.g. "rerun+http://127.0.0.1:9876/proxy". If it ends in ".rrd", save to that file.
  /// @return True on success.
  bool init(const std::string & applicationId = "okvis2", const std::string & connectUrl = "");

  /// @brief Log the static camera rig (extrinsics and pinhole intrinsics).
  /// @param cameras The camera system.
  void logCameras(const cameras::NCameraSystem & cameras);

  /// @brief Estimator update; signature matches ViInterface::OptimisedGraphCallback.
  /// @param latestState Newest state.
  /// @param trackingState Tracking state of the newest state.
  /// @param updatedStates All states changed by this optimisation (incl. loop closures).
  /// @param landmarks Current landmarks.
  void logEstimatorUpdate(const State & latestState, const TrackingState & trackingState,
                          std::shared_ptr<const AlignedMap<StateId, State>> updatedStates,
                          std::shared_ptr<const MapPointVector> landmarks);

  /// @brief Add a ground-truth pose T_WS in the ground-truth world frame. Thread-safe; may be called
  ///        ahead of the estimator (e.g. from a dataset reader) -- only poses up to the latest
  ///        estimated state are drawn.
  /// @param t Timestamp.
  /// @param T_WS Ground-truth pose.
  void addGroundTruthPose(const Time & t, const kinematics::Transformation & T_WS);

  /**
   * @brief Log one submap's surface mesh under world/submaps/<id>.
   *
   * Logged statically and keyed by submap id, so re-logging after a loop closure moves the mesh in
   * place instead of piling up copies.
   * @param id The submap id.
   * @param vertices Triangle soup in world coordinates: three consecutive vertices per triangle.
   * @param colours Per-vertex RGB in [0,1]; must be empty or the same length as vertices.
   */
  void logSubmapMesh(uint64_t id,
                     const std::vector<Eigen::Vector3f> & vertices,
                     const std::vector<Eigen::Vector3f> & colours = std::vector<Eigen::Vector3f>());

  /// @brief True if init() succeeded.
  bool enabled() const { return enabled_; }

 private:
#ifdef OKVIS_USE_RERUN
  /// @brief Align (once) and log the ground-truth path up to the given state's time.
  /// @param latestState The newest estimated state.
  void logGroundTruth(const State & latestState);

  std::unique_ptr<rerun::RecordingStream> rec_; ///< The recording stream.
  std::mutex mutex_; ///< Protects the members below.
  /// Trajectory bookkeeping as used by the ROS publisher: non-keyframe states are stored relative
  /// to their keyframe, so they follow when loop closures move the keyframes.
  Trajectory trajectory_;
  StateId lastTrajectoryLogId_; ///< State ID when the trajectory was last logged.
  std::mutex groundTruthMutex_; ///< Protects groundTruth_.
  AlignedMap<uint64_t, kinematics::Transformation> groundTruth_; ///< Ground truth by time [ns].
  bool groundTruthAligned_ = false; ///< Has the ground-truth alignment been computed?
  kinematics::Transformation T_WWgt_; ///< Estimate world from ground-truth world.
#endif
  bool enabled_ = false; ///< Is logging active?
};

}  // namespace okvis

#endif  // INCLUDE_OKVIS_RERUNVISUALIZER_HPP_
