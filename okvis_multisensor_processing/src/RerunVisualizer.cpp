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
 * @file RerunVisualizer.cpp
 * @brief Source file for the RerunVisualizer class.
 */

#include <okvis/RerunVisualizer.hpp>

#include <glog/logging.h>

#ifdef OKVIS_USE_RERUN
#include <algorithm>
#include <Eigen/Geometry>
#include <cmath>
#include <iterator>
#include <set>
#include <vector>

#include <rerun.hpp>

#include <okvis/cameras/PinholeCamera.hpp>
#endif

namespace okvis {

#ifdef OKVIS_USE_RERUN

namespace {

const char* kTimeline = "sensor_time";

// Re-log the full trajectory at most this often (in states), plus on every loop closure.
const uint64_t kTrajectoryLogEveryNStates = 10;

// Keep ground-truth poses at most at this rate, to bound what is re-logged.
const uint64_t kGroundTruthMinSpacingNs = 50000000; // 20 Hz

// Only align ground truth if a pose exists this close to the first estimated state.
const uint64_t kGroundTruthMaxAlignGapNs = 100000000; // 0.1 s

rerun::Transform3D toRerun(const kinematics::Transformation & T) {
  const Eigen::Vector3d r = T.r();
  const Eigen::Quaterniond q = T.q();
  return rerun::Transform3D::from_translation(
             {float(r.x()), float(r.y()), float(r.z())})
      .with_quaternion(rerun::Quaternion::from_xyzw(
          float(q.x()), float(q.y()), float(q.z()), float(q.w())));
}

}  // namespace

RerunVisualizer::RerunVisualizer() = default;

RerunVisualizer::~RerunVisualizer() {
  if (rec_) {
    (void)rec_->flush_blocking(2.0f);
  }
}

bool RerunVisualizer::available() {
  return true;
}

bool RerunVisualizer::init(const std::string & applicationId, const std::string & connectUrl) {
  rec_ = std::make_unique<rerun::RecordingStream>(applicationId);
  rerun::Error error;
  const bool toFile = connectUrl.size() > 4 && connectUrl.substr(connectUrl.size() - 4) == ".rrd";
  if (connectUrl.empty()) {
    error = rec_->spawn();
  } else if (toFile) {
    error = rec_->save(connectUrl);
  } else {
    error = rec_->connect_grpc(connectUrl);
  }
  if (error.is_err()) {
    LOG(ERROR) << "Rerun: could not " << (connectUrl.empty() ? "spawn viewer" : "open " + connectUrl)
               << ": " << error.description
               << (connectUrl.empty() ? " (is `rerun` in PATH? pip install rerun-sdk==" RERUN_SDK_HEADER_VERSION ")" : "");
    rec_.reset();
    return false;
  }
  rec_->log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);
  enabled_ = true;
  LOG(INFO) << "Rerun: logging to "
            << (connectUrl.empty() ? std::string("spawned viewer") : connectUrl);
  return true;
}

void RerunVisualizer::logCameras(const cameras::NCameraSystem & cameras) {
  if (!enabled_) {
    return;
  }
  for (size_t i = 0; i < cameras.numCameras(); ++i) {
    const std::string entity = "world/body/cam" + std::to_string(i);
    rec_->log_static(entity, toRerun(*cameras.T_SC(i)));
    const auto geometry = cameras.cameraGeometry(i);
    const auto pinhole = std::dynamic_pointer_cast<const cameras::PinholeCameraBase>(geometry);
    if (pinhole) {
      const std::array<float, 9> K_columns = {
          float(pinhole->focalLengthU()), 0.0f, 0.0f,
          0.0f, float(pinhole->focalLengthV()), 0.0f,
          float(pinhole->imageCenterU()), float(pinhole->imageCenterV()), 1.0f};
      rec_->log_static(entity,
                       rerun::Pinhole(rerun::components::PinholeProjection(K_columns))
                           .with_resolution(float(geometry->imageWidth()),
                                            float(geometry->imageHeight()))
                           .with_camera_xyz(rerun::components::ViewCoordinates::RDF)
                           .with_image_plane_distance(0.3f));
    }
  }
}

void RerunVisualizer::logEstimatorUpdate(
    const State & latestState, const TrackingState & trackingState,
    std::shared_ptr<const AlignedMap<StateId, State>> updatedStates,
    std::shared_ptr<const MapPointVector> landmarks) {
  if (!enabled_) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const uint64_t tNs = latestState.timestamp.toNSec();
  rec_->set_time_duration_nanos(kTimeline, int64_t(tNs));

  // current pose
  rec_->log("world/body", toRerun(latestState.T_WS));

  // trajectory. Only keyframes (and the window) are re-optimised and reported in updatedStates;
  // okvis::Trajectory re-derives every non-keyframe pose from its keyframe, so the whole path
  // follows loop closures instead of leaving old frames at their pre-correction positions.
  std::set<StateId> affectedStateIds;
  if (updatedStates && updatedStates->count(trackingState.id)) {
    trajectory_.update(trackingState, updatedStates, affectedStateIds);
  }
  const bool loopClosed = trackingState.recognisedPlace
      || (updatedStates && updatedStates->size() > 20);
  if (loopClosed || !lastTrajectoryLogId_.isInitialised()
      || latestState.id.value() >= lastTrajectoryLogId_.value() + kTrajectoryLogEveryNStates) {
    std::vector<rerun::Vec3D> strip;
    strip.reserve(trajectory_.stateIds().size());
    State state;
    for (const auto & id : trajectory_.stateIds()) {
      if (!trajectory_.getState(id, state)) {
        continue;
      }
      const Eigen::Vector3d r = state.T_WS.r();
      strip.push_back({float(r.x()), float(r.y()), float(r.z())});
    }
    // static: only the latest trajectory is kept, so memory does not grow with every update
    rec_->log_static("world/trajectory",
                     rerun::LineStrips3D(std::vector<rerun::LineStrip3D>{rerun::LineStrip3D(strip)})
                         .with_colors(rerun::Color(0, 200, 0))
                         .with_radii(0.02f));
    // keyframes, at their current (possibly loop-closed) positions
    std::vector<rerun::Position3D> keyframes;
    for (const auto & keyframe : trajectory_.keyframeStates()) {
      const Eigen::Vector3d r = keyframe.second.T_WS.r();
      keyframes.emplace_back(float(r.x()), float(r.y()), float(r.z()));
    }
    rec_->log_static("world/keyframes", rerun::Points3D(keyframes)
                                            .with_colors(rerun::Color(255, 160, 0))
                                            .with_radii(0.05f));
    lastTrajectoryLogId_ = latestState.id;
    logGroundTruth(latestState);
  }


  // landmarks
  if (landmarks) {
    std::vector<rerun::Position3D> points;
    std::vector<rerun::Color> colours;
    points.reserve(landmarks->size());
    colours.reserve(landmarks->size());
    for (const auto & lm : *landmarks) {
      const double w = lm.point[3];
      if (std::fabs(w) < 1.0e-6) {
        continue; // at infinity
      }
      points.emplace_back(float(lm.point[0] / w), float(lm.point[1] / w), float(lm.point[2] / w));
      const uint8_t q = uint8_t(std::min(1.0, std::max(0.0, lm.quality)) * 255.0);
      colours.emplace_back(uint8_t(255 - q), q, 80);
    }
    rec_->log_static("world/landmarks",
                     rerun::Points3D(points).with_colors(colours).with_radii(0.015f));
  }
}

void RerunVisualizer::addGroundTruthPose(const Time & t, const kinematics::Transformation & T_WS) {
  if (!enabled_) {
    return;
  }
  const uint64_t tNs = t.toNSec();
  std::lock_guard<std::mutex> lock(groundTruthMutex_);
  if (!groundTruth_.empty() && tNs < groundTruth_.rbegin()->first + kGroundTruthMinSpacingNs) {
    return;
  }
  groundTruth_[tNs] = T_WS;
}

void RerunVisualizer::logGroundTruth(const State & latestState) {
  const uint64_t tNs = latestState.timestamp.toNSec();
  std::vector<rerun::Vec3D> strip;
  {
    std::lock_guard<std::mutex> lock(groundTruthMutex_);
    if (groundTruth_.empty()) {
      return;
    }
    if (!groundTruthAligned_) {
      // ground-truth pose closest in time to this (first) state
      auto it = groundTruth_.lower_bound(tNs);
      if (it == groundTruth_.end()) {
        it = std::prev(it);
      } else if (it != groundTruth_.begin() && tNs - std::prev(it)->first < it->first - tNs) {
        it = std::prev(it);
      }
      const uint64_t gap = it->first > tNs ? it->first - tNs : tNs - it->first;
      if (gap > kGroundTruthMaxAlignGapNs) {
        return; // no ground truth near this state yet
      }
      // both worlds are gravity-aligned: keep only the yaw of T_W_Wgt = T_WS_est * T_WS_gt^-1
      const kinematics::Transformation T_WWgt = latestState.T_WS * it->second.inverse();
      const Eigen::Matrix3d C = T_WWgt.C();
      const Eigen::Quaterniond q_yaw(
          Eigen::AngleAxisd(std::atan2(C(1, 0), C(0, 0)), Eigen::Vector3d::UnitZ()));
      T_WWgt_ = kinematics::Transformation(latestState.T_WS.r() - q_yaw * it->second.r(), q_yaw);
      groundTruthAligned_ = true;
    }
    for (auto it = groundTruth_.begin(); it != groundTruth_.end() && it->first <= tNs; ++it) {
      const Eigen::Vector3d p = T_WWgt_.C() * it->second.r() + T_WWgt_.r();
      strip.push_back({float(p.x()), float(p.y()), float(p.z())});
    }
  }
  if (strip.size() < 2) {
    return;
  }
  rec_->log_static("world/ground_truth",
                   rerun::LineStrips3D(std::vector<rerun::LineStrip3D>{rerun::LineStrip3D(strip)})
                       .with_colors(rerun::Color(220, 50, 50))
                       .with_radii(0.02f));
}

void RerunVisualizer::logSubmapMesh(uint64_t id, const std::vector<Eigen::Vector3f> & vertices,
                                    const std::vector<Eigen::Vector3f> & colours) {
  if (!enabled_ || vertices.empty()) {
    return;
  }
  std::vector<rerun::Position3D> positions;
  positions.reserve(vertices.size());
  for (const auto & vertex : vertices) {
    positions.emplace_back(vertex.x(), vertex.y(), vertex.z());
  }
  // The vertices are a triangle soup, so face normals are cheap. Without normals AND without
  // colours a Mesh3D renders as a flat white blob with no visible geometry, so when the map has no
  // colour we shade the triangles by their normal instead.
  const bool haveColours = colours.size() == vertices.size();
  std::vector<rerun::Vector3D> normals(vertices.size(), rerun::Vector3D(0.0f, 0.0f, 1.0f));
  std::vector<rerun::Color> vertexColours;
  auto to8 = [](float c) { return uint8_t(std::min(1.0f, std::max(0.0f, c)) * 255.0f); };
  if (haveColours) {
    vertexColours.reserve(colours.size());
    for (const auto & colour : colours) {
      vertexColours.emplace_back(to8(colour.x()), to8(colour.y()), to8(colour.z()));
    }
  } else {
    vertexColours.assign(vertices.size(), rerun::Color(200, 200, 200));
  }
  const Eigen::Vector3f light = Eigen::Vector3f(0.3f, 0.4f, 0.9f).normalized();
  for (size_t i = 0; i + 2 < vertices.size(); i += 3) {
    const Eigen::Vector3f edge0 = vertices[i + 1] - vertices[i];
    const Eigen::Vector3f edge1 = vertices[i + 2] - vertices[i];
    Eigen::Vector3f normal = edge0.cross(edge1);
    const float length = normal.norm();
    normal = length > 1.0e-9f ? Eigen::Vector3f(normal / length) : Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    // two-sided: marching cubes winding is not guaranteed to face the viewer
    const float shade = 0.35f + 0.65f * std::fabs(normal.dot(light));
    for (size_t j = 0; j < 3; ++j) {
      normals[i + j] = rerun::Vector3D(normal.x(), normal.y(), normal.z());
      if (!haveColours) {
        vertexColours[i + j] = rerun::Color(to8(shade * 0.92f), to8(shade * 0.95f), to8(shade));
      }
    }
  }

  const std::string entity = "world/submaps/" + std::to_string(id);
  std::lock_guard<std::mutex> lock(mutex_);
  rec_->log_static(entity, rerun::Mesh3D(positions)
                               .with_vertex_normals(normals)
                               .with_vertex_colors(vertexColours));
}

#else  // OKVIS_USE_RERUN

RerunVisualizer::RerunVisualizer() = default;
RerunVisualizer::~RerunVisualizer() = default;

bool RerunVisualizer::available() {
  return false;
}

bool RerunVisualizer::init(const std::string &, const std::string &) {
  LOG(WARNING) << "Rerun requested, but OKVIS was built without it (configure with -DUSE_RERUN=ON).";
  return false;
}

void RerunVisualizer::logCameras(const cameras::NCameraSystem &) {}

void RerunVisualizer::logEstimatorUpdate(const State &, const TrackingState &,
                                         std::shared_ptr<const AlignedMap<StateId, State>>,
                                         std::shared_ptr<const MapPointVector>) {}

void RerunVisualizer::addGroundTruthPose(const Time &, const kinematics::Transformation &) {}

void RerunVisualizer::logSubmapMesh(uint64_t, const std::vector<Eigen::Vector3f> &,
                                    const std::vector<Eigen::Vector3f> &) {}

#endif  // OKVIS_USE_RERUN

}  // namespace okvis
