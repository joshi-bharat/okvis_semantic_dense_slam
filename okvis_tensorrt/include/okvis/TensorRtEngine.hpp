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
 * @file TensorRtEngine.hpp
 * @brief Header file for the TensorRtEngine class.
 */

#ifndef OKVIS_TENSORRT_ENGINE_HPP
#define OKVIS_TENSORRT_ENGINE_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}

namespace okvis {

/**
 * @brief Minimal, framework-free TensorRT runtime for a serialised engine (.engine / .plan).
 *
 * Enumerates the engine's I/O tensors, resolves dynamic input shapes, owns the device buffers and
 * a CUDA stream, and exchanges data with the host as cv::Mat. Not thread-safe: use one instance
 * per inference thread.
 *
 * Build an engine from ONNX with e.g.
 *   trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
 * The engine is specific to the GPU model and TensorRT version it was built with.
 */
class TensorRtEngine {
 public:
  /// @brief Element type of an I/O tensor.
  enum class DataType { Float32, Float16, Int8, Int32, Int64, UInt8, Bool, Other };

  /// @brief Description of one I/O tensor.
  struct TensorInfo {
    std::string name; ///< Tensor name.
    bool isInput = false; ///< Input (true) or output (false).
    DataType dataType = DataType::Other; ///< Element type.
    std::vector<int64_t> shape; ///< Shape; -1 for dynamic dimensions of inputs before setInputShape.
  };

  /// @brief Deserialise an engine. Throws std::runtime_error on failure.
  /// @param enginePath Path to the serialised engine.
  /// @param device CUDA device index.
  explicit TensorRtEngine(const std::string & enginePath, int device = 0);

  /// @brief Frees device memory and TensorRT objects.
  ~TensorRtEngine();

  TensorRtEngine(const TensorRtEngine &) = delete;
  TensorRtEngine & operator=(const TensorRtEngine &) = delete;

  /// @brief All inputs, in engine order.
  std::vector<TensorInfo> inputs() const;

  /// @brief All outputs, in engine order.
  std::vector<TensorInfo> outputs() const;

  /// @brief Current (resolved) shape of a tensor.
  std::vector<int64_t> shape(const std::string & name) const;

  /// @brief Set the shape of a dynamic input. Must be called before setInput for dynamic inputs.
  /// @return False if the shape is outside the engine's optimisation profile.
  bool setInputShape(const std::string & name, const std::vector<int64_t> & shape);

  /// @brief Copy host data to an input. The Mat must be continuous and hold exactly the tensor's
  ///        number of elements with the tensor's element type (any cv shape / channel layout).
  void setInput(const std::string & name, const cv::Mat & data);

  /// @brief Run inference synchronously.
  void infer();

  /// @brief Copy an output to the host as a continuous Mat with the output's element count
  ///        (rows = 1, cols = element count, depth = element type).
  cv::Mat getOutput(const std::string & name);

  /// @brief Size in bytes of one element of the given type.
  static size_t elementSize(DataType type);

  /// @brief OpenCV depth (CV_32F, ...) for the given type, or -1 if unsupported.
  static int cvDepth(DataType type);

  /// @brief Human-readable summary of all I/O tensors.
  std::string describe() const;

 private:
  /// @brief (Re)allocate the device buffer of a tensor to fit its current shape.
  void ensureBuffer(const std::string & name);

  struct Buffer {
    void* device = nullptr; ///< Device pointer.
    size_t bytes = 0; ///< Allocated size.
  };

  class Logger;
  std::unique_ptr<Logger> logger_; ///< TensorRT logger (forwards to glog).
  std::unique_ptr<nvinfer1::IRuntime> runtime_; ///< Runtime.
  std::unique_ptr<nvinfer1::ICudaEngine> engine_; ///< Engine.
  std::unique_ptr<nvinfer1::IExecutionContext> context_; ///< Execution context.
  void* stream_ = nullptr; ///< CUDA stream (cudaStream_t).
  int device_ = 0; ///< CUDA device.
  std::vector<std::string> names_; ///< I/O tensor names in engine order.
  std::map<std::string, Buffer> buffers_; ///< Device buffers by tensor name.
};

}  // namespace okvis

#endif  // OKVIS_TENSORRT_ENGINE_HPP
