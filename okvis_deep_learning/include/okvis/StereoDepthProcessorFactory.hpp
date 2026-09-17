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
 * @file StereoDepthProcessorFactory.hpp
 * @brief Chooses the stereo depth backend (TensorRT engine or TorchScript model).
 */

#ifndef OKVIS_STEREO_DEPTH_PROCESSOR_FACTORY_HPP
#define OKVIS_STEREO_DEPTH_PROCESSOR_FACTORY_HPP

#include <fstream>
#include <stdexcept>
#include <string>

#include <glog/logging.h>

#include <okvis/DeepLearningProcessor.hpp>
#include <okvis/Parameters.hpp>

#ifdef OKVIS_USE_TENSORRT
#include <okvis/TensorRtStereo2DepthProcessor.hpp>
#endif
#ifdef OKVIS_USE_NN
#include <okvis/Stereo2DepthProcessor.hpp>
#endif

namespace okvis {

/**
 * @brief Create the stereo depth processor for the models in modelDir.
 *
 * Preference order:
 *  1. tensorrt_parameters/engine from the config, if set (requires USE_TENSORRT);
 *  2. TensorRT engine modelDir/depth-model.engine (or .plan), if built with USE_TENSORRT;
 *  3. TorchScript model modelDir/depth-model.pt, if built with USE_NN.
 * @param parameters OKVIS parameters.
 * @param modelDir Directory holding the models (the resources directory).
 * @return The processor (caller owns it). Throws if no usable backend/model exists.
 */
inline DeepLearningProcessor* createStereoDepthProcessor(ViParameters & parameters,
                                                         const std::string & modelDir) {
  const std::string & configured = parameters.tensorrt.engine;
  if (!configured.empty()) {
#ifdef OKVIS_USE_TENSORRT
    if (!std::ifstream(configured).good()) {
      throw std::runtime_error("tensorrt_parameters/engine not found: " + configured);
    }
    LOG(INFO) << "Stereo depth backend: TensorRT (" << configured << ", from config)";
    return new TensorRtStereo2DepthProcessor(parameters, configured);
#else
    throw std::runtime_error("tensorrt_parameters/engine is set (" + configured
                             + "), but OKVIS was built without USE_TENSORRT");
#endif
  }
#ifdef OKVIS_USE_TENSORRT
  for (const char * extension : {".engine", ".plan"}) {
    const std::string enginePath = modelDir + "/depth-model" + extension;
    if (std::ifstream(enginePath).good()) {
      LOG(INFO) << "Stereo depth backend: TensorRT (" << enginePath << ")";
      return new TensorRtStereo2DepthProcessor(parameters, enginePath);
    }
  }
#endif
#ifdef OKVIS_USE_NN
  LOG(INFO) << "Stereo depth backend: TorchScript (" << modelDir << "/depth-model.pt)";
  return new Stereo2DepthProcessor(parameters, modelDir);
#else
  throw std::runtime_error("no stereo depth model: expected " + modelDir
                           + "/depth-model.engine (TensorRT). Build with USE_NN=ON to use"
                             " depth-model.pt instead.");
#endif
}

}  // namespace okvis

#endif  // OKVIS_STEREO_DEPTH_PROCESSOR_FACTORY_HPP
