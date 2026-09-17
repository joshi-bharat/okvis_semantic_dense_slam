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
 * @file TumVieDatasetReader.cpp
 * @brief Source file for the TumVieDatasetReader class.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <Eigen/Core>

#include <okvis/TumVieDatasetReader.hpp>

namespace okvis {

// Stereo images are hardware-triggered; anything further apart than this is a broken pair.
static const Duration kStereoTolerance(0.001);

// At power-up the cameras' auto-exposure alternates between ~19 ms (saturated) and 0.1 ms (black)
// frames. Leading frames up to the last one exposed shorter than this, within the first
// kExposureBurstMaxFrames frames, are skipped.
static const double kExposureBurstMinUs = 1000.0;
static const size_t kExposureBurstMaxFrames = 200;

TumVieDatasetReader::TumVieDatasetReader(const std::string & path, const Duration & deltaT,
                                         int numCameras, const Duration & imuLookahead) :
  deltaT_(deltaT), imuLookahead_(imuLookahead), numCameras_(numCameras) {
  streaming_ = false;
  setDatasetPath(path);
  counter_ = 0;
}

TumVieDatasetReader::~TumVieDatasetReader() {
  stopStreaming();
}

bool TumVieDatasetReader::setDatasetPath(const std::string & path) {
  path_ = path;
  return true;
}

bool TumVieDatasetReader::setStartingDelay(const Duration &deltaT)
{
  if(streaming_) {
    LOG(WARNING)<< "starting delay ignored, because streaming already started";
    return false;
  }
  deltaT_ = deltaT;
  return true;
}

bool TumVieDatasetReader::isStreaming()
{
  return streaming_;
}

double TumVieDatasetReader::completion() const {
  if(streaming_ && numImages_ > 0) {
    return double(counter_)/double(numImages_);
  }
  return 0.0;
}

Time TumVieDatasetReader::timeFromMicroseconds(double us) {
  // Doubles represent these values (< 1e12 ns) exactly enough for rounding to the nanosecond.
  Time t;
  t.fromNSec(uint64_t(std::llround(us * 1000.0)));
  return t;
}

bool TumVieDatasetReader::readTimestamps(const std::string & filename, std::vector<Time> & times) {
  std::ifstream file(filename);
  if(!file.good()) {
    return false;
  }
  times.clear();
  std::string line;
  while (std::getline(file, line)) {
    if(line.empty() || line[0] == '#') {
      continue;
    }
    times.push_back(timeFromMicroseconds(std::stod(line)));
  }
  return true;
}

bool TumVieDatasetReader::readImuSample(ImuSample & sample) {
  std::string line;
  while (std::getline(imuFile_, line)) {
    if(line.empty() || line[0] == '#') {
      continue;
    }
    std::stringstream stream(line);
    double us;
    stream >> us >> sample.gyr[0] >> sample.gyr[1] >> sample.gyr[2]
           >> sample.acc[0] >> sample.acc[1] >> sample.acc[2]; // trailing temperature ignored
    OKVIS_ASSERT_TRUE(Exception, !stream.fail(), "malformed IMU line: " << line)
    sample.t = timeFromMicroseconds(us);
    return true;
  }
  return false;
}

bool TumVieDatasetReader::startStreaming() {
  OKVIS_ASSERT_TRUE(Exception, !imagesCallbacks_.empty(), "no add image callback registered")
  OKVIS_ASSERT_TRUE(Exception, !imuCallbacks_.empty(), "no add IMU callback registered")
  OKVIS_ASSERT_TRUE(Exception, numCameras_ == -1 || numCameras_ == 1 || numCameras_ == 2,
                    "TUM-VIE provides 1 or 2 cameras, requested " << numCameras_)

  // open the IMU file
  imuFile_.open(path_ + "/imu_data.txt");
  OKVIS_ASSERT_TRUE(Exception, imuFile_.good(), "no imu file found at " << path_+"/imu_data.txt")

  // camera timestamps
  const int numCameras = numCameras_ == -1 ? 2 : numCameras_;
  const std::vector<std::string> sides = {"left", "right"};
  imageTimes_.clear();
  imageDirs_.clear();
  for (int i = 0; i < numCameras; ++i) {
    const std::string dir = path_ + "/" + sides[size_t(i)] + "_images";
    const std::string tsFile = dir + "/image_timestamps_" + sides[size_t(i)] + ".txt";
    std::vector<Time> times;
    OKVIS_ASSERT_TRUE(Exception, readTimestamps(tsFile, times), "no timestamps at " << tsFile)
    OKVIS_ASSERT_TRUE(Exception, !times.empty(), "no images listed in " << tsFile)
    LOG(INFO) << "No. cam " << i << " (" << sides[size_t(i)] << ") images: " << times.size();
    imageTimes_.push_back(times);
    imageDirs_.push_back(dir);
  }
  if(numCameras == 2) {
    OKVIS_ASSERT_TRUE(Exception, imageTimes_[0].size() == imageTimes_[1].size(),
                      "left/right image count mismatch: " << imageTimes_[0].size() << " vs "
                      << imageTimes_[1].size())
  }
  numImages_ = imageTimes_[0].size();

  // skip the auto-exposure start-up burst, if exposure times are available
  firstImage_ = 0;
  for (int i = 0; i < numCameras; ++i) {
    const std::string expFile = imageDirs_[size_t(i)] + "/image_exposures_" + sides[size_t(i)] + ".txt";
    std::ifstream file(expFile);
    if (!file.good()) {
      continue;
    }
    std::string line;
    size_t k = 0;
    while (std::getline(file, line) && k < std::min(numImages_, kExposureBurstMaxFrames)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      if (std::stod(line) < kExposureBurstMinUs) {
        firstImage_ = std::max(firstImage_, k + 1);
      }
      ++k;
    }
  }
  if (firstImage_ > 0) {
    LOG(INFO) << "Skipping " << firstImage_ << " frames of auto-exposure start-up burst.";
  }

  counter_ = 0;
  streaming_ = true;
  processingThread_ = std::thread(&TumVieDatasetReader::processing, this);

  return true;
}

bool TumVieDatasetReader::stopStreaming() {
  // Stop the pipeline
  if(processingThread_.joinable()) {
    processingThread_.join();
    streaming_ = false;
  }
  return true;
}

void TumVieDatasetReader::processing() {
  const size_t numCameras = imageTimes_.size();
  const Time start = imageTimes_[0].front();

  ImuSample imu;
  bool haveImu = readImuSample(imu);
  OKVIS_ASSERT_TRUE(Exception, haveImu, "no IMU measurements in " << path_ << "/imu_data.txt")
  const Time firstImuTime = imu.t;

  size_t numSkippedNoImu = 0;
  char filename[32];
  for (size_t k = 0; k < numImages_ && streaming_; ++k) {
    const Time t = imageTimes_[0][k];

    // stream all IMU measurements up to slightly past this image
    while (haveImu && imu.t <= t + imuLookahead_) {
      if (imu.t - start + okvis::Duration(1.0) > deltaT_) {
        for (auto &imuCallback : imuCallbacks_) {
          imuCallback(imu.t, imu.acc, imu.gyr);
        }
      }
      haveImu = readImuSample(imu);
    }
    if (!haveImu && imu.t < t + imuLookahead_) {
      LOG(INFO) << "IMU data exhausted at t=" << imu.t << "; stopping.";
      break;
    }

    // skip exposure burst, requested initial duration, and frames the IMU does not cover yet
    if (k < firstImage_ || t - start <= deltaT_) {
      ++counter_;
      continue;
    }
    if (t < firstImuTime + imuLookahead_) {
      ++numSkippedNoImu;
      ++counter_;
      continue;
    }

    std::map<size_t, cv::Mat> images;
    for (size_t i = 0; i < numCameras; ++i) {
      if (i > 0) {
        OKVIS_ASSERT_TRUE(Exception,
                          std::fabs((imageTimes_[i][k] - t).toSec()) < kStereoTolerance.toSec(),
                          "unsynchronised stereo pair at index " << k)
      }
      std::snprintf(filename, sizeof(filename), "/%05zu.jpg", k);
      const std::string file = imageDirs_[i] + filename;
      cv::Mat image = cv::imread(file, cv::IMREAD_GRAYSCALE);
      OKVIS_ASSERT_TRUE(Exception, !image.empty(), "cam " << i << " missing image: " << file)
      images[i] = image;
    }

    for (auto &imagesCallback : imagesCallbacks_) {
      imagesCallback(t, images, std::map<size_t, cv::Mat>());
    }
    ++counter_;
  }

  if (numSkippedNoImu > 0) {
    LOG(INFO) << "Skipped " << numSkippedNoImu << " initial frames without IMU coverage.";
  }
  streaming_ = false;
}

}
