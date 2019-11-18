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
 * Copyright (c) 2015 by Contributors
 * \file cuda_utils.h
 * \brief Common CUDA utilities.
 */
#ifndef MXNET_COMMON_CUDA_UTILS_H_
#define MXNET_COMMON_CUDA_UTILS_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/optional.h>
#include <mshadow/base.h>
#include <mxnet/libinfo.h>

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

#define QUOTE(x) #x
#define QUOTEVALUE(x) QUOTE(x)

#if MXNET_USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <vector>

#define STATIC_ASSERT_CUDA_VERSION_GE(min_version) \
  static_assert(CUDA_VERSION >= min_version, "Compiled-against CUDA version " \
      QUOTEVALUE(CUDA_VERSION) " is too old, please upgrade system to version " \
      QUOTEVALUE(min_version) " or later.")

/*!
 * \brief When compiling a __device__ function, check that the architecture is >= Kepler (3.0)
 *        Note that __CUDA_ARCH__ is not defined outside of a __device__ function
 */
#ifdef __CUDACC__
inline __device__ bool __is_supported_cuda_architecture() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
#error "Fermi and earlier GPU architectures are not supported (architecture versions less than 3.0)"
  return false;
#else
  return true;
#endif  // __CUDA_ARCH__ < 300
}
#endif  // __CUDACC__

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
        << "cuBLAS: " << mxnet::common::cuda::CublasGetErrorString(e); \
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
        << "cuSolver: " << mxnet::common::cuda::CusolverGetErrorString(e); \
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
        << "cuRAND: " << mxnet::common::cuda::CurandGetErrorString(e); \
  }

/*!
 * \brief Protected NVRTC call.
 * \param func Expression to call.
 *
 * It checks for NVRTC errors after invocation of the expression.
 */
#define NVRTC_CALL(x)                                   \
  {                                                     \
    nvrtcResult result = x;                             \
    CHECK_EQ(result, NVRTC_SUCCESS)                     \
      << #x " failed with error "                       \
      << nvrtcGetErrorString(result);                   \
  }

/*!
 * \brief Protected CUDA driver call.
 * \param func Expression to call.
 *
 * It checks for CUDA driver errors after invocation of the expression.
 */
#define CUDA_DRIVER_CALL(func)                                          \
  {                                                                     \
    CUresult e = (func);                                                \
    if (e != CUDA_SUCCESS) {                                            \
      char const * err_msg = nullptr;                                         \
      if (cuGetErrorString(e, &err_msg) == CUDA_ERROR_INVALID_VALUE) {  \
        LOG(FATAL) << "CUDA Driver: Unknown error " << e;               \
      } else {                                                          \
        LOG(FATAL) << "CUDA Driver: " << err_msg;                       \
      }                                                                 \
    }                                                                   \
  }


#if !defined(_MSC_VER)
#define CUDA_UNROLL _Pragma("unroll")
#define CUDA_NOUNROLL _Pragma("nounroll")
#else
#define CUDA_UNROLL
#define CUDA_NOUNROLL
#endif

namespace mxnet {
namespace common {
/*! \brief common utils for cuda */
namespace cuda {
/*!
 * \brief Converts between C++ datatypes and enums/constants needed by cuBLAS.
 */
template<typename DType>
struct CublasType;

// With CUDA v8, cuBLAS adopted use of cudaDataType_t instead of its own
// datatype cublasDataType_t.  The older cudaDataType_t values could be
// included below, but since this class was introduced to support the cuBLAS v8
// call cublasGemmEx(), burdening the class with the legacy type values
// was not needed.

template<>
struct CublasType<float> {
  static const int kFlag = mshadow::kFloat32;
#if CUDA_VERSION >= 8000
  static const cudaDataType_t kCudaFlag = CUDA_R_32F;
#endif
  typedef float ScaleType;
  static const float one;
  static const float zero;
};
template<>
struct CublasType<double> {
  static const int kFlag = mshadow::kFloat64;
#if CUDA_VERSION >= 8000
  static const cudaDataType_t kCudaFlag = CUDA_R_64F;
#endif
  typedef double ScaleType;
  static const double one;
  static const double zero;
};
template<>
struct CublasType<mshadow::half::half_t> {
  static const int kFlag = mshadow::kFloat16;
#if CUDA_VERSION >= 8000
  static const cudaDataType_t kCudaFlag = CUDA_R_16F;
#endif
  typedef float ScaleType;
  static const mshadow::half::half_t one;
  static const mshadow::half::half_t zero;
};
template<>
struct CublasType<uint8_t> {
  static const int kFlag = mshadow::kUint8;
#if CUDA_VERSION >= 8000
  static const cudaDataType_t kCudaFlag = CUDA_R_8I;
#endif
  typedef uint8_t ScaleType;
  static const uint8_t one = 1;
  static const uint8_t zero = 0;
};
template<>
struct CublasType<int32_t> {
  static const int kFlag = mshadow::kInt32;
#if CUDA_VERSION >= 8000
  static const cudaDataType_t kCudaFlag = CUDA_R_32I;
#endif
  typedef int32_t ScaleType;
  static const int32_t one = 1;
  static const int32_t zero = 0;
};

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

template <typename DType>
inline DType __device__ CudaMax(DType a, DType b) {
    return a > b ? a : b;
}

template <typename DType>
inline DType __device__ CudaMin(DType a, DType b) {
    return a < b ? a : b;
}

class DeviceStore {
 public:
  /*! \brief default constructor- only optionally restores previous device */
  explicit DeviceStore(int requested_device = -1, bool restore = true) :
    restore_device_(-1),
    current_device_(requested_device),
    restore_(restore) {
    if (restore_)
      CUDA_CALL(cudaGetDevice(&restore_device_));
    if (requested_device != restore_device_) {
      SetDevice(requested_device);
    }
  }

  ~DeviceStore() {
    if (restore_ &&
        current_device_ != restore_device_ &&
        current_device_ != -1 &&
        restore_device_ != -1)
      CUDA_CALL(cudaSetDevice(restore_device_));
  }

  void SetDevice(int device) {
    if (device != -1) {
      CUDA_CALL(cudaSetDevice(device));
      current_device_ = device;
    }
  }

 private:
  int restore_device_;
  int current_device_;
  bool restore_;
};

/*!
 * \brief Get the largest datatype suitable to read
 *         requested number of bytes.
 *
 *  \input Number of bytes to be read
 *  \return mshadow representation of type that could
 *          be used for reading
 */
int get_load_type(size_t N);

/*!
 * \brief Determine how many rows in a 2D matrix should a block
 *        of threads handle based on the row size and the number
 *        of threads in a block.
 * \param row_size Size of the row expressed in the number of reads required to fully
 *                 load it. For example, if the row has N elements, but  each thread
 *                 reads 2 elements with a single read, row_size should be N / 2.
 * \param num_threads_per_block Number of threads in a block.
 * \return the number of rows that should be handled by a single block.
 */
int get_rows_per_block(size_t row_size, int num_threads_per_block);

}  // namespace cuda
}  // namespace common
}  // namespace mxnet

/*! \brief Maximum number of GPUs */
constexpr size_t kMaxNumGpus = 64;

// The implementations below assume that accesses of 32-bit ints are inherently atomic and
// can be read/written by multiple threads without locks.  The values held should be < 2^31.

/*!
 * \brief Return an attribute GPU `device_id`.
 * \param device_id The device index of the cuda-capable gpu of interest.
 * \param cached_values An array of attributes for already-looked-up GPUs.
 * \param attr The attribute, by number.
 * \param attr_name A string representation of the attribute, for error messages.
 * \return the gpu's attribute value.
 */
inline int cudaAttributeLookup(int device_id, std::vector<int32_t> *cached_values,
                               cudaDeviceAttr attr, const char *attr_name) {
  if (device_id < 0 || device_id >= static_cast<int>(cached_values->size())) {
    LOG(FATAL) << attr_name << "(device_id) called with invalid id: " << device_id;
  } else if ((*cached_values)[device_id] < 0) {
    int temp = -1;
    CUDA_CALL(cudaDeviceGetAttribute(&temp, attr, device_id));
    (*cached_values)[device_id] = static_cast<int32_t>(temp);
  }
  return (*cached_values)[device_id];
}

/*!
 * \brief Determine major version number of the gpu's cuda compute architecture.
 * \param device_id The device index of the cuda-capable gpu of interest.
 * \return the major version number of the gpu's cuda compute architecture.
 */
inline int ComputeCapabilityMajor(int device_id) {
  static std::vector<int32_t> capability_major(kMaxNumGpus, -1);
  return cudaAttributeLookup(device_id, &capability_major,
                             cudaDevAttrComputeCapabilityMajor, "ComputeCapabilityMajor");
}

/*!
 * \brief Determine minor version number of the gpu's cuda compute architecture.
 * \param device_id The device index of the cuda-capable gpu of interest.
 * \return the minor version number of the gpu's cuda compute architecture.
 */
inline int ComputeCapabilityMinor(int device_id) {
  static std::vector<int32_t> capability_minor(kMaxNumGpus, -1);
  return cudaAttributeLookup(device_id, &capability_minor,
                             cudaDevAttrComputeCapabilityMinor, "ComputeCapabilityMinor");
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
 * \brief Return the number of streaming multiprocessors of GPU `device_id`.
 * \param device_id The device index of the cuda-capable gpu of interest.
 * \return the gpu's count of streaming multiprocessors.
 */
inline int MultiprocessorCount(int device_id) {
  static std::vector<int32_t> sm_counts(kMaxNumGpus, -1);
  return cudaAttributeLookup(device_id, &sm_counts,
                             cudaDevAttrMultiProcessorCount, "MultiprocessorCount");
}

/*!
 * \brief Return the shared memory size in bytes of each of the GPU's streaming multiprocessors.
 * \param device_id The device index of the cuda-capable gpu of interest.
 * \return the shared memory size per streaming multiprocessor.
 */
inline int MaxSharedMemoryPerMultiprocessor(int device_id) {
  static std::vector<int32_t> max_smem_per_mutiprocessor(kMaxNumGpus, -1);
  return cudaAttributeLookup(device_id, &max_smem_per_mutiprocessor,
                             cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                             "MaxSharedMemoryPerMultiprocessor");
}

/*!
 * \brief Return whether the GPU `device_id` supports cooperative-group kernel launching.
 * \param device_id The device index of the cuda-capable gpu of interest.
 * \return the gpu's ability to run cooperative-group kernels.
 */
inline bool SupportsCooperativeLaunch(int device_id) {
  static std::vector<int32_t> coop_launch(kMaxNumGpus, -1);
  return cudaAttributeLookup(device_id, &coop_launch,
                             cudaDevAttrCooperativeLaunch, "SupportsCooperativeLaunch");
}

/*!
 * \brief Determine whether a cuda-capable gpu's architecture supports float16 math.
 *        Assume not if device_id is negative.
 * \param device_id The device index of the cuda-capable gpu of interest.
 * \return whether the gpu's architecture supports float16 math.
 */
inline bool SupportsFloat16Compute(int device_id) {
  if (device_id < 0) {
    return false;
  } else {
    // Kepler and most Maxwell GPUs do not support fp16 compute
    int computeCapabilityMajor = ComputeCapabilityMajor(device_id);
    return (computeCapabilityMajor > 5) ||
           (computeCapabilityMajor == 5 && ComputeCapabilityMinor(device_id) >= 3);
  }
}

/*!
 * \brief Determine whether a cuda-capable gpu's architecture supports Tensor Core math.
 *        Assume not if device_id is negative.
 * \param device_id The device index of the cuda-capable gpu of interest.
 * \return whether the gpu's architecture supports Tensor Core math.
 */
inline bool SupportsTensorCore(int device_id) {
  // Volta (sm_70) supports TensorCore algos
  return device_id >= 0 &&
         ComputeCapabilityMajor(device_id) >=7;
}

// The policy if the user hasn't set the environment variable MXNET_CUDA_ALLOW_TENSOR_CORE
#define MXNET_CUDA_ALLOW_TENSOR_CORE_DEFAULT true

/*!
 * \brief Returns global policy for TensorCore algo use.
 * \return whether to allow TensorCore algo (if not specified by the Operator locally).
 */
inline bool GetEnvAllowTensorCore() {
  // Since these statics are in the '.h' file, they will exist and will be set
  // separately in each compilation unit.  Not ideal, but cleaner than creating a
  // cuda_utils.cc solely to have a single instance and initialization.
  static bool allow_tensor_core = false;
  static bool is_set = false;
  if (!is_set) {
    // Use of optional<bool> here permits: "0", "1", "true" and "false" to all be legal.
    bool default_value = MXNET_CUDA_ALLOW_TENSOR_CORE_DEFAULT;
    allow_tensor_core = dmlc::GetEnv("MXNET_CUDA_ALLOW_TENSOR_CORE",
                                     dmlc::optional<bool>(default_value)).value();
    is_set = true;
  }
  return allow_tensor_core;
}

// The policy if the user hasn't set the environment variable
// CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
#define MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION_DEFAULT false

/*!
 * \brief Returns global policy for TensorCore implicit type casting
 */
inline bool GetEnvAllowTensorCoreConversion() {
  // Use of optional<bool> here permits: "0", "1", "true" and "false" to all be
  // legal.
  bool default_value = MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION_DEFAULT;
  return dmlc::GetEnv("MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION",
                      dmlc::optional<bool>(default_value))
      .value();
}

#if CUDA_VERSION >= 9000
// Sets the cuBLAS math mode that determines the 'allow TensorCore' policy.  Returns previous.
inline cublasMath_t SetCublasMathMode(cublasHandle_t blas_handle, cublasMath_t new_math_type) {
  auto handle_math_mode = CUBLAS_DEFAULT_MATH;
  CUBLAS_CALL(cublasGetMathMode(blas_handle, &handle_math_mode));
  CUBLAS_CALL(cublasSetMathMode(blas_handle, new_math_type));
  return handle_math_mode;
}
#endif

#endif  // MXNET_USE_CUDA

#if MXNET_USE_CUDNN

#include <cudnn.h>

// Creating CUDNN_VERSION_AS_STRING as follows avoids a static_assert error message that shows
// the formula for CUDNN_VERSION, i.e. "1000 * 7 + 100 * 6 + 0" rather than number "7600".
static_assert(CUDNN_PATCHLEVEL < 100 && CUDNN_MINOR < 10,
              "CUDNN_VERSION_AS_STRING macro assumptions violated.");
#if CUDNN_PATCHLEVEL >= 10
#define CUDNN_VERSION_AS_STRING QUOTEVALUE(CUDNN_MAJOR) \
                                QUOTEVALUE(CUDNN_MINOR) \
                                QUOTEVALUE(CUDNN_PATCHLEVEL)
#else
#define CUDNN_VERSION_AS_STRING QUOTEVALUE(CUDNN_MAJOR) \
                                QUOTEVALUE(CUDNN_MINOR) \
                                "0" QUOTEVALUE(CUDNN_PATCHLEVEL)
#endif

#define STATIC_ASSERT_CUDNN_VERSION_GE(min_version) \
  static_assert(CUDNN_VERSION >= min_version, "Compiled-against cuDNN version " \
      CUDNN_VERSION_AS_STRING " is too old, please upgrade system to version " \
      QUOTEVALUE(min_version) " or later.")

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
  STATIC_ASSERT_CUDNN_VERSION_GE(7000);
  int max_algos = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn_handle, &max_algos));
  return max_algos;
}

/*!
 * \brief Return max number of perf structs cudnnFindConvolutionBackwardFilterAlgorithm()
 *        may want to populate.
 * \param cudnn_handle cudnn handle needed to perform the inquiry.
 * \return max number of perf structs cudnnFindConvolutionBackwardFilterAlgorithm() may
 *         want to populate.
 */
inline int MaxBackwardFilterAlgos(cudnnHandle_t cudnn_handle) {
  STATIC_ASSERT_CUDNN_VERSION_GE(7000);
  int max_algos = 0;
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnn_handle, &max_algos));
  return max_algos;
}

/*!
 * \brief Return max number of perf structs cudnnFindConvolutionBackwardDataAlgorithm()
 *        may want to populate.
 * \param cudnn_handle cudnn handle needed to perform the inquiry.
 * \return max number of perf structs cudnnFindConvolutionBackwardDataAlgorithm() may
 *         want to populate.
 */
inline int MaxBackwardDataAlgos(cudnnHandle_t cudnn_handle) {
  STATIC_ASSERT_CUDNN_VERSION_GE(7000);
  int max_algos = 0;
  CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnn_handle, &max_algos));
  return max_algos;
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
#ifdef __CUDACC__
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

static inline __device__ void atomicAdd(uint8_t *address, uint8_t val) {
  unsigned int * address_as_ui = (unsigned int *) (address - ((size_t)address & 0x3));
  unsigned int old = *address_as_ui;
  unsigned int shift = (((size_t)address & 0x3) << 3);
  unsigned int sum;
  unsigned int assumed;

  do {
    assumed = old;
    sum = val + static_cast<uint8_t>((old >> shift) & 0xff);
    old = (old & ~(0x000000ff << shift)) | (sum << shift);
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}

static inline __device__ void atomicAdd(int8_t *address, int8_t val) {
  unsigned int * address_as_ui = (unsigned int *) (address - ((size_t)address & 0x3));
  unsigned int old = *address_as_ui;
  unsigned int shift = (((size_t)address & 0x3) << 3);
  unsigned int sum;
  unsigned int assumed;

  do {
    assumed = old;
    sum = val + static_cast<int8_t>((old >> shift) & 0xff);
    old = (old & ~(0x000000ff << shift)) | (sum << shift);
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}

// Overload atomicAdd to work for signed int64 on all architectures
static inline  __device__  void atomicAdd(int64_t *address, int64_t val) {
  atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(val)); // NOLINT
}

template <typename DType>
__device__ inline DType ldg(const DType* address) {
#if __CUDA_ARCH__ >= 350
    return __ldg(address);
#else
    return *address;
#endif
}

template <typename OP, typename T>
__device__ inline T warp_reduce(T value, OP redfun) {
  value = redfun(value, __shfl_down_sync(0xffffffff, value, 16));
  value = redfun(value, __shfl_down_sync(0xffffffff, value, 8));
  value = redfun(value, __shfl_down_sync(0xffffffff, value, 4));
  value = redfun(value, __shfl_down_sync(0xffffffff, value, 2));
  value = redfun(value, __shfl_down_sync(0xffffffff, value, 1));
  return value;
}

template <typename OP>
__device__ inline mshadow::half::half_t warp_reduce(mshadow::half::half_t value, OP redfun) {
  float v = static_cast<float>(value);
  v = redfun(v, __shfl_down_sync(0xffffffff, v, 16));
  v = redfun(v, __shfl_down_sync(0xffffffff, v, 8));
  v = redfun(v, __shfl_down_sync(0xffffffff, v, 4));
  v = redfun(v, __shfl_down_sync(0xffffffff, v, 2));
  v = redfun(v, __shfl_down_sync(0xffffffff, v, 1));
  return mshadow::half::half_t(v);
}

#endif  // __CUDACC__

#endif  // MXNET_COMMON_CUDA_UTILS_H_
