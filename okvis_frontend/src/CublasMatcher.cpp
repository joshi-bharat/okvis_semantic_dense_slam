/**
 * This file is part of the SuperPoint frontend for OKVIS2-X.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#include <okvis/CublasMatcher.hpp>

#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace okvis {

namespace {
void check(cudaError_t status, const char * what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CublasMatcher: ") + what + ": "
                             + cudaGetErrorString(status));
  }
}
void check(cublasStatus_t status, const char * what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("CublasMatcher: ") + what + " failed");
  }
}

/// @brief Device buffer that grows on demand and is never shrunk.
struct Buffer {
  void * data = nullptr;
  size_t bytes = 0;
  void reserve(size_t required) {
    if (required <= bytes) {
      return;
    }
    if (data) {
      cudaFree(data);
    }
    check(cudaMalloc(&data, required), "cudaMalloc");
    bytes = required;
  }
  ~Buffer() {
    if (data) {
      cudaFree(data);
    }
  }
};
} // namespace

struct CublasMatcher::Impl {
  cublasHandle_t handle = nullptr;
  Buffer query;
  Buffer train;
  Buffer scores;
};

CublasMatcher::CublasMatcher() : impl_(new Impl) {
  check(cublasCreate(&impl_->handle), "cublasCreate");
  // Half inputs accumulate in fp32, which keeps 256-term dot products accurate.
  check(cublasSetMathMode(impl_->handle, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
}

CublasMatcher::~CublasMatcher() {
  if (impl_->handle) {
    cublasDestroy(impl_->handle);
  }
}

cv::Mat CublasMatcher::score(const cv::Mat & query, const cv::Mat & train) {
  CV_Assert(query.type() == CV_16F && train.type() == CV_16F);
  CV_Assert(query.cols == train.cols);
  CV_Assert(query.isContinuous() && train.isContinuous());
  const int n = query.rows;
  const int m = train.rows;
  const int d = query.cols;
  if (n == 0 || m == 0) {
    return cv::Mat(n, m, CV_32F);
  }

  const size_t queryBytes = size_t(n) * size_t(d) * sizeof(uint16_t);
  const size_t trainBytes = size_t(m) * size_t(d) * sizeof(uint16_t);
  const size_t scoreBytes = size_t(n) * size_t(m) * sizeof(float);
  impl_->query.reserve(queryBytes);
  impl_->train.reserve(trainBytes);
  impl_->scores.reserve(scoreBytes);

  check(cudaMemcpy(impl_->query.data, query.data, queryBytes, cudaMemcpyHostToDevice),
        "cudaMemcpy query");
  check(cudaMemcpy(impl_->train.data, train.data, trainBytes, cudaMemcpyHostToDevice),
        "cudaMemcpy train");

  // Both matrices are row-major D-column blocks. cuBLAS is column-major, so a row-major
  // (n x d) matrix is a column-major (d x n) one; scores = query * train^T therefore becomes
  // C(m x n) = train(d x m)^T * query(d x n), read back as row-major (n x m).
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check(cublasGemmEx(impl_->handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, d, &alpha,
                     impl_->train.data, CUDA_R_16F, d,
                     impl_->query.data, CUDA_R_16F, d, &beta,
                     impl_->scores.data, CUDA_R_32F, m,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx");

  cv::Mat scores(n, m, CV_32F);
  check(cudaMemcpy(scores.data, impl_->scores.data, scoreBytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy scores");
  return scores;
}

} // namespace okvis
