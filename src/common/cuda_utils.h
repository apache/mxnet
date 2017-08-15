/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file cuda_utils.h
 * \brief CUDA debugging utilities.
 */
#ifndef MXNET_COMMON_CUDA_UTILS_H_
#define MXNET_COMMON_CUDA_UTILS_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/optional.h>
#include <mshadow/base.h>

/*! \brief Macros/inlines to assist CLion to parse Cuda files (*.cu, *.cuh) */
#ifdef __JETBRAINS_IDE__
#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __forceinline__
#define __shared__
inline void __syncthreads() {}
inline void __threadfence_block() {}
template<class T> inline T __clz(const T val) { return val; }
struct __cuda_fake_struct { int x; int y; int z; };
extern __cuda_fake_struct blockDim;
extern __cuda_fake_struct threadIdx;
extern __cuda_fake_struct blockIdx;
#endif

#if MXNET_USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

namespace mxnet {
namespace common {
/*! \brief common utils for cuda */
namespace cuda {
/*!
 * \brief Converts between C++ datatypes and constants needed by cuBLAS.
 */
template<typename DType>
struct CublasType;

template<>
struct CublasType<float> {
  static const float one;
  static const float zero;
};
template<>
struct CublasType<double> {
  static const double one;
  static const double zero;
};
template<>
struct CublasType<mshadow::half::half_t> {
  static const mshadow::half::half_t one;
  static const mshadow::half::half_t zero;
};
template<>
struct CublasType<uint8_t> {
  static const uint8_t one = 1;
  static const uint8_t zero = 0;
};
template<>
struct CublasType<int32_t> {
  static const int32_t one = 1;
  static const int32_t zero = 0;
};

// Return a pointer to the value '1' in the format required by `algo_precision`.
// Used typically for the 'alpha' parameter to the GEMM kernel.
inline const void *one(int algo_precision) {
  const void *answer = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(algo_precision, CType, {
    answer = &CublasType<CType>::one;
  })
  return answer;
}

// Return a pointer to the value '0' in the format required by `algo_precision`.
// Used typically for the 'beta' parameter to the GEMM kernel.
inline const void *zero(int algo_precision) {
  const void *answer = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(algo_precision, CType, {
    answer = &CublasType<CType>::zero;
  })
  return answer;
}

/*!
 * \brief Get string representation of cuBLAS errors.
 * \param error The error.
 * \return String representation.
 */
inline const char* CublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  default:
    break;
  }
  return "Unknown cuBLAS status";
}

#if CUDA_VERSION >= 8000
/*!
 * \brief Create the proper constant for indicating cuBLAS transposition, if desired.
 * \param transpose Whether transposition should be performed.
 * \return the yes/no transposition-indicating constant.
 */
inline cublasOperation_t CublasTransposeOp(bool transpose) {
  return transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
}
#endif

/*!
 * \brief Get string representation of cuSOLVER errors.
 * \param error The error.
 * \return String representation.
 */
inline const char* CusolverGetErrorString(cusolverStatus_t error) {
  switch (error) {
  case CUSOLVER_STATUS_SUCCESS:
    return "CUSOLVER_STATUS_SUCCESS";
  case CUSOLVER_STATUS_NOT_INITIALIZED:
    return "CUSOLVER_STATUS_NOT_INITIALIZED";
  case CUSOLVER_STATUS_ALLOC_FAILED:
    return "CUSOLVER_STATUS_ALLOC_FAILED";
  case CUSOLVER_STATUS_INVALID_VALUE:
    return "CUSOLVER_STATUS_INVALID_VALUE";
  case CUSOLVER_STATUS_ARCH_MISMATCH:
    return "CUSOLVER_STATUS_ARCH_MISMATCH";
  case CUSOLVER_STATUS_EXECUTION_FAILED:
    return "CUSOLVER_STATUS_EXECUTION_FAILED";
  case CUSOLVER_STATUS_INTERNAL_ERROR:
    return "CUSOLVER_STATUS_INTERNAL_ERROR";
  case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  default:
    break;
  }
  return "Unknown cuSOLVER status";
}

/*!
 * \brief Get string representation of cuRAND errors.
 * \param status The status.
 * \return String representation.
 */
inline const char* CurandGetErrorString(curandStatus_t status) {
  switch (status) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown cuRAND status";
}

}  // namespace cuda
}  // namespace common
}  // namespace mxnet

/*!
 * \brief Check CUDA error.
 * \param msg Message to print if an error occured.
 */
#define CHECK_CUDA_ERROR(msg)                                                \
  {                                                                          \
    cudaError_t e = cudaGetLastError();                                      \
    CHECK_EQ(e, cudaSuccess) << (msg) << " CUDA: " << cudaGetErrorString(e); \
  }

/*!
 * \brief Protected CUDA call.
 * \param func Expression to call.
 *
 * It checks for CUDA errors after invocation of the expression.
 */
#define CUDA_CALL(func)                                            \
  {                                                                \
    cudaError_t e = (func);                                        \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

/*!
 * \brief Protected cuBLAS call.
 * \param func Expression to call.
 *
 * It checks for cuBLAS errors after invocation of the expression.
 */
#define CUBLAS_CALL(func)                                       \
  {                                                             \
    cublasStatus_t e = (func);                                  \
    CHECK_EQ(e, CUBLAS_STATUS_SUCCESS)                          \
        << "cuBLAS: " << common::cuda::CublasGetErrorString(e); \
  }

/*!
 * \brief Protected cuSolver call.
 * \param func Expression to call.
 *
 * It checks for cuSolver errors after invocation of the expression.
 */
#define CUSOLVER_CALL(func)                                         \
  {                                                                 \
    cusolverStatus_t e = (func);                                    \
    CHECK_EQ(e, CUSOLVER_STATUS_SUCCESS)                            \
        << "cuSolver: " << common::cuda::CusolverGetErrorString(e); \
  }

/*!
 * \brief Protected cuRAND call.
 * \param func Expression to call.
 *
 * It checks for cuRAND errors after invocation of the expression.
 */
#define CURAND_CALL(func)                                       \
  {                                                             \
    curandStatus_t e = (func);                                  \
    CHECK_EQ(e, CURAND_STATUS_SUCCESS)                          \
        << "cuRAND: " << common::cuda::CurandGetErrorString(e); \
  }

/*!
 * \brief Determine major version number of the gpu's cuda compute architecture.
 * \param device_id The device index of the cuda-capable gpu of interest.
 * \return the major version number of the gpu's cuda compute architecture.
 */
inline int ComputeCapabilityMajor(int device_id) {
  int major = 0;
  CUDA_CALL(cudaDeviceGetAttribute(&major,
                                   cudaDevAttrComputeCapabilityMajor, device_id));
  return major;
}

/*!
 * \brief Determine minor version number of the gpu's cuda compute architecture.
 * \param device_id The device index of the cuda-capable gpu of interest.
 * \return the minor version number of the gpu's cuda compute architecture.
 */
inline int ComputeCapabilityMinor(int device_id) {
  int minor = 0;
  CUDA_CALL(cudaDeviceGetAttribute(&minor,
                                   cudaDevAttrComputeCapabilityMinor, device_id));
  return minor;
}

/*!
 * \brief Return the integer SM architecture (e.g. Volta = 70).
 * \param device_id The device index of the cuda-capable gpu of interest.
 * \return the gpu's cuda compute architecture as an int.
 */
inline int SMArch(int device_id) {
  auto major = ComputeCapabilityMajor(device_id);
  auto minor = ComputeCapabilityMinor(device_id);
  return 10 * major + minor;
}

/*!
 * \brief Determine whether a cuda-capable gpu's architecture supports float16 math.
 * \param device_id The device index of the cuda-capable gpu of interest.
 * \return whether the gpu's architecture supports float16 math.
 */
inline bool SupportsFloat16Compute(int device_id) {
  // Kepler and most Maxwell GPUs do not support fp16 compute
  int computeCapabilityMajor = ComputeCapabilityMajor(device_id);
  int computeCapabilityMinor = ComputeCapabilityMinor(device_id);
  return (computeCapabilityMajor > 5) ||
      (computeCapabilityMajor == 5 && computeCapabilityMinor >= 3);
}

/*!
 * \brief Determine whether a cuda-capable gpu's architecture supports Tensor Core math.
 * \param device_id The device index of the cuda-capable gpu of interest.
 * \return whether the gpu's architecture supports Tensor Core math.
 */
inline bool SupportsTensorCore(int device_id) {
  // Volta (sm_70) supports TensorCore algos
  int computeCapabilityMajor = ComputeCapabilityMajor(device_id);
  return (computeCapabilityMajor >= 7);
}

// The policy if the user hasn't set the environment variable MXNET_CUDA_ALLOW_TENSOR_CORE
#define MXNET_CUDA_ALLOW_TENSOR_CORE_DEFAULT true

/*!
 * \brief Returns global policy for TensorCore algo use.
 * \return whether to allow TensorCore algo (if not specified by the Operator locally).
 */
inline bool GetEnvAllowTensorCore() {
  // Use of optional<bool> here permits: "0", "1", "true" and "false" to all be legal.
  bool default_value = MXNET_CUDA_ALLOW_TENSOR_CORE_DEFAULT;
  return dmlc::GetEnv("MXNET_CUDA_ALLOW_TENSOR_CORE",
                      dmlc::optional<bool>(default_value)).value();
}

#endif  // MXNET_USE_CUDA

#if MXNET_USE_CUDNN

#include <cudnn.h>

#define CUDNN_CALL(func)                                                      \
  {                                                                           \
    cudnnStatus_t e = (func);                                                 \
    CHECK_EQ(e, CUDNN_STATUS_SUCCESS) << "cuDNN: " << cudnnGetErrorString(e); \
  }

/*!
 * \brief Return max number of perf structs cudnnFindConvolutionForwardAlgorithm()
 *        may want to populate.
 * \param cudnn_handle cudnn handle needed to perform the inquiry.
 * \return max number of perf structs cudnnFindConvolutionForwardAlgorithm() may
 *         want to populate.
 */
inline int MaxForwardAlgos(cudnnHandle_t cudnn_handle) {
#if CUDNN_MAJOR >= 7
  int max_algos = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn_handle, &max_algos));
  return max_algos;
#else
  return 10;
#endif
}

/*!
 * \brief Return max number of perf structs cudnnFindConvolutionBackwardFilterAlgorithm()
 *        may want to populate.
 * \param cudnn_handle cudnn handle needed to perform the inquiry.
 * \return max number of perf structs cudnnFindConvolutionBackwardFilterAlgorithm() may
 *         want to populate.
 */
inline int MaxBackwardFilterAlgos(cudnnHandle_t cudnn_handle) {
#if CUDNN_MAJOR >= 7
  int max_algos = 0;
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnn_handle, &max_algos));
  return max_algos;
#else
  return 10;
#endif
}

/*!
 * \brief Return max number of perf structs cudnnFindConvolutionBackwardDataAlgorithm()
 *        may want to populate.
 * \param cudnn_handle cudnn handle needed to perform the inquiry.
 * \return max number of perf structs cudnnFindConvolutionBackwardDataAlgorithm() may
 *         want to populate.
 */
inline int MaxBackwardDataAlgos(cudnnHandle_t cudnn_handle) {
#if CUDNN_MAJOR >= 7
  int max_algos = 0;
  CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnn_handle, &max_algos));
  return max_algos;
#else
  return 10;
#endif
}

#endif  // MXNET_USE_CUDNN

// Overload atomicAdd to work for floats on all architectures
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
// From CUDA Programming Guide
static inline  __device__  void atomicAdd(double *address, double val) {
  unsigned long long* address_as_ull =                  // NOLINT(*)
    reinterpret_cast<unsigned long long*>(address);     // NOLINT(*)
  unsigned long long old = *address_as_ull;             // NOLINT(*)
  unsigned long long assumed;                           // NOLINT(*)

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                    __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
}
#endif

// Overload atomicAdd for half precision
// Taken from:
// https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh
#if defined(__CUDA_ARCH__)
static inline __device__ void atomicAdd(mshadow::half::half_t *address,
                                        mshadow::half::half_t val) {
  unsigned int *address_as_ui =
      reinterpret_cast<unsigned int *>(reinterpret_cast<char *>(address) -
                                   (reinterpret_cast<size_t>(address) & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    mshadow::half::half_t hsum;
    hsum.half_ =
        reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
    hsum += val;
    old = reinterpret_cast<size_t>(address) & 2
              ? (old & 0xffff) | (hsum.half_ << 16)
              : (old & 0xffff0000) | hsum.half_;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}
#endif

#endif  // MXNET_COMMON_CUDA_UTILS_H_
