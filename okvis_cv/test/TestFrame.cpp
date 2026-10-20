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
 * @file TestFrame.cpp
 * @brief Runs Frame tests.
 * @author Stefan Leutenegger
 */

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <brisk/brisk.h>
#pragma GCC diagnostic pop

#include "okvis/cameras/PinholeCamera.hpp"
#include "okvis/cameras/NoDistortion.hpp"
#include "okvis/cameras/RadialTangentialDistortion.hpp"
#include "okvis/cameras/EquidistantDistortion.hpp"
#include "okvis/Frame.hpp"
#include "okvis/internal/Network.hpp"

TEST(Frame, functions)
{

  // instantiate all possible versions of test cameras
  std::vector<std::shared_ptr<okvis::cameras::CameraBase> > cameras;
  cameras.push_back(
      okvis::cameras::PinholeCamera<okvis::cameras::NoDistortion>::createTestObject());
  cameras.push_back(
      okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion>::createTestObject());
  cameras.push_back(
      okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>::createTestObject());

  for (size_t c = 0; c < cameras.size(); ++c) {

#ifdef __ARM_NEON__
   std::shared_ptr<cv::FeatureDetector> detector(
        new brisk::BriskFeatureDetector(34, 2));
#else
   std::shared_ptr<cv::FeatureDetector> detector(
        new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(
            34, 2, 800, 450));
#endif
 
    std::shared_ptr<cv::DescriptorExtractor> extractor(
        new cv::BriskDescriptorExtractor(true, false));

    // create a stupid random image
    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> eigenImage(752,480);
    eigenImage.setRandom();
    cv::Mat image(480, 752, CV_8UC1, eigenImage.data());
    std::shared_ptr<okvis::Network> network(new okvis::Network());
    okvis::Frame frame(image, cameras.at(c), detector, extractor, network);

    // run
    frame.detect();
    frame.describe();
  }
}


TEST(Frame, addressesFp16DescriptorRowsByBytes)
{
  // SuperPoint descriptors: 256 fp16 elements, i.e. 512 bytes per row.
  okvis::Frame frame;
  std::vector<cv::KeyPoint> keypoints(3, cv::KeyPoint(cv::Point2f(1.0f, 1.0f), 8.0f));
  frame.resetKeypoints(keypoints);
  cv::Mat descriptors(3, 256, CV_16F, cv::Scalar(0));
  frame.resetDescriptors(descriptors);

  EXPECT_EQ(frame.descriptorDim(), 256);
  EXPECT_EQ(frame.descriptorType(), CV_16F);
  EXPECT_EQ(frame.descriptors().data, descriptors.data);
  EXPECT_EQ(frame.keypointDescriptor(1), descriptors.ptr<uchar>(1));
  EXPECT_EQ(frame.keypointDescriptor(2), descriptors.ptr<uchar>(2));
}

TEST(Frame, addressesBriskDescriptorRowsAsBefore)
{
  okvis::Frame frame;
  std::vector<cv::KeyPoint> keypoints(2, cv::KeyPoint(cv::Point2f(1.0f, 1.0f), 8.0f));
  frame.resetKeypoints(keypoints);
  cv::Mat descriptors(2, 48, CV_8UC1, cv::Scalar(0));
  frame.resetDescriptors(descriptors);

  EXPECT_EQ(frame.descriptorDim(), 48);
  EXPECT_EQ(frame.descriptorType(), CV_8UC1);
  EXPECT_EQ(frame.keypointDescriptor(1), descriptors.data + 48);
}
