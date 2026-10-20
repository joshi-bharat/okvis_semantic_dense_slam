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
 * @file TensorRtEngine.cpp
 * @brief Source file for the TensorRtEngine class.
 */

#include <okvis/TensorRtEngine.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <glog/logging.h>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace okvis {

namespace {

void checkCuda(cudaError_t status, const char * what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(status));
  }
}

TensorRtEngine::DataType fromTrt(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT: return TensorRtEngine::DataType::Float32;
    case nvinfer1::DataType::kHALF: return TensorRtEngine::DataType::Float16;
    case nvinfer1::DataType::kINT8: return TensorRtEngine::DataType::Int8;
    case nvinfer1::DataType::kINT32: return TensorRtEngine::DataType::Int32;
    case nvinfer1::DataType::kINT64: return TensorRtEngine::DataType::Int64;
    case nvinfer1::DataType::kUINT8: return TensorRtEngine::DataType::UInt8;
    case nvinfer1::DataType::kBOOL: return TensorRtEngine::DataType::Bool;
    default: return TensorRtEngine::DataType::Other;
  }
}

std::vector<int64_t> fromDims(const nvinfer1::Dims & dims) {
  return std::vector<int64_t>(dims.d, dims.d + dims.nbDims);
}

int64_t volume(const std::vector<int64_t> & shape) {
  int64_t v = 1;
  for (const auto d : shape) {
    if (d < 0) {
      return -1;
    }
    v *= d;
  }
  return v;
}

std::string shapeString(const std::vector<int64_t> & shape) {
  std::stringstream s;
  s << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    s << (i ? "," : "") << shape[i];
  }
  s << "]";
  return s.str();
}

const char * typeName(TensorRtEngine::DataType type) {
  switch (type) {
    case TensorRtEngine::DataType::Float32: return "float32";
    case TensorRtEngine::DataType::Float16: return "float16";
    case TensorRtEngine::DataType::Int8: return "int8";
    case TensorRtEngine::DataType::Int32: return "int32";
    case TensorRtEngine::DataType::Int64: return "int64";
    case TensorRtEngine::DataType::UInt8: return "uint8";
    case TensorRtEngine::DataType::Bool: return "bool";
    default: return "other";
  }
}

}  // namespace

class TensorRtEngine::Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char * msg) noexcept override {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR: LOG(ERROR) << "TensorRT: " << msg; break;
      case Severity::kWARNING: LOG(WARNING) << "TensorRT: " << msg; break;
      case Severity::kINFO: VLOG(1) << "TensorRT: " << msg; break;
      default: VLOG(2) << "TensorRT: " << msg; break;
    }
  }
};

TensorRtEngine::TensorRtEngine(const std::string & enginePath, int device)
    : logger_(new Logger), device_(device) {
  checkCuda(cudaSetDevice(device_), "cudaSetDevice");

  std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    throw std::runtime_error("TensorRT engine not found: " + enginePath);
  }
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> blob(static_cast<size_t>(size));
  if (!file.read(blob.data(), size)) {
    throw std::runtime_error("could not read TensorRT engine " + enginePath);
  }

  runtime_.reset(nvinfer1::createInferRuntime(*logger_));
  if (!runtime_) {
    throw std::runtime_error("could not create TensorRT runtime");
  }
  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine_) {
    throw std::runtime_error("could not deserialise TensorRT engine " + enginePath
                             + " (built for a different GPU or TensorRT version?)");
  }
  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    throw std::runtime_error("could not create TensorRT execution context");
  }
  cudaStream_t stream = nullptr;
  checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  stream_ = stream;

  for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
    names_.emplace_back(engine_->getIOTensorName(i));
  }
  LOG(INFO) << "Loaded TensorRT engine " << enginePath << ":\n" << describe();
}

TensorRtEngine::~TensorRtEngine() {
  cudaSetDevice(device_);
  if (stream_) {
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
  }
  // destroy TensorRT objects before freeing the memory they reference
  context_.reset();
  engine_.reset();
  runtime_.reset();
  for (auto & buffer : buffers_) {
    cudaFree(buffer.second.device);
  }
  if (stream_) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
  }
}

std::vector<TensorRtEngine::TensorInfo> TensorRtEngine::inputs() const {
  std::vector<TensorInfo> result;
  for (const auto & name : names_) {
    if (engine_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT) {
      result.push_back({name, true, fromTrt(engine_->getTensorDataType(name.c_str())), shape(name)});
    }
  }
  return result;
}

std::vector<TensorRtEngine::TensorInfo> TensorRtEngine::outputs() const {
  std::vector<TensorInfo> result;
  for (const auto & name : names_) {
    if (engine_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kOUTPUT) {
      result.push_back({name, false, fromTrt(engine_->getTensorDataType(name.c_str())), shape(name)});
    }
  }
  return result;
}

std::vector<int64_t> TensorRtEngine::shape(const std::string & name) const {
  return fromDims(context_->getTensorShape(name.c_str()));
}

bool TensorRtEngine::setInputShape(const std::string & name, const std::vector<int64_t> & shape) {
  nvinfer1::Dims dims;
  dims.nbDims = int32_t(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    dims.d[i] = shape[i];
  }
  return context_->setInputShape(name.c_str(), dims);
}

size_t TensorRtEngine::elementSize(DataType type) {
  switch (type) {
    case DataType::Float32: return 4;
    case DataType::Float16: return 2;
    case DataType::Int8: return 1;
    case DataType::Int32: return 4;
    case DataType::Int64: return 8;
    case DataType::UInt8: return 1;
    case DataType::Bool: return 1;
    default: return 0;
  }
}

int TensorRtEngine::cvDepth(DataType type) {
  switch (type) {
    case DataType::Float32: return CV_32F;
    case DataType::Float16: return CV_16F;
    case DataType::Int8: return CV_8S;
    case DataType::Int32: return CV_32S;
    case DataType::UInt8: return CV_8U;
    case DataType::Bool: return CV_8U;
    default: return -1;
  }
}

void TensorRtEngine::ensureBuffer(const std::string & name) {
  const std::vector<int64_t> s = shape(name);
  const int64_t n = volume(s);
  if (n < 0) {
    throw std::runtime_error("TensorRT tensor " + name + " has unresolved shape " + shapeString(s)
                             + " -- set all dynamic input shapes first");
  }
  const size_t bytes =
      size_t(n) * elementSize(fromTrt(engine_->getTensorDataType(name.c_str())));
  Buffer & buffer = buffers_[name];
  if (buffer.bytes < bytes) {
    if (buffer.device) {
      checkCuda(cudaFree(buffer.device), "cudaFree");
    }
    checkCuda(cudaMalloc(&buffer.device, bytes), "cudaMalloc");
    buffer.bytes = bytes;
  }
  if (!context_->setTensorAddress(name.c_str(), buffer.device)) {
    throw std::runtime_error("could not bind TensorRT tensor " + name);
  }
}

void TensorRtEngine::setInput(const std::string & name, const cv::Mat & data) {
  cudaSetDevice(device_);
  ensureBuffer(name);
  const DataType type = fromTrt(engine_->getTensorDataType(name.c_str()));
  const size_t bytes = size_t(volume(shape(name))) * elementSize(type);
  if (!data.isContinuous() || data.total() * data.elemSize() != bytes
      || CV_MAT_DEPTH(data.type()) != cvDepth(type)) {
    throw std::runtime_error("input " + name + ": expected " + std::to_string(bytes) + " bytes of "
                             + typeName(type) + " for shape " + shapeString(shape(name)));
  }
  checkCuda(cudaMemcpy(buffers_.at(name).device, data.data, bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy (input)");
}

void TensorRtEngine::infer() {
  cudaSetDevice(device_);
  for (const auto & name : names_) {
    if (engine_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kOUTPUT) {
      ensureBuffer(name); // output shapes are known once all input shapes are set
    }
  }
  if (!context_->enqueueV3(static_cast<cudaStream_t>(stream_))) {
    throw std::runtime_error("TensorRT inference failed");
  }
  checkCuda(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)), "cudaStreamSynchronize");
}

cv::Mat TensorRtEngine::getOutput(const std::string & name) {
  cudaSetDevice(device_);
  const DataType type = fromTrt(engine_->getTensorDataType(name.c_str()));
  const int depth = cvDepth(type);
  if (depth < 0) {
    throw std::runtime_error(std::string("output ") + name + " has unsupported type "
                             + typeName(type));
  }
  const int64_t n = volume(shape(name));
  cv::Mat result(1, int(n), depth);
  checkCuda(cudaMemcpy(result.data, buffers_.at(name).device, size_t(n) * elementSize(type),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy (output)");
  return result;
}

std::string TensorRtEngine::describe() const {
  std::stringstream s;
  for (const auto & name : names_) {
    const bool input = engine_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
    s << "  " << (input ? "input  " : "output ") << name << " "
      << typeName(fromTrt(engine_->getTensorDataType(name.c_str()))) << " "
      << shapeString(fromDims(engine_->getTensorShape(name.c_str()))) << "\n";
  }
  return s.str();
}

}  // namespace okvis
