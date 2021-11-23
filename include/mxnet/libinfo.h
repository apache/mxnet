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
 * \file libinfo.h
 * \author larroy
 * \brief get features of the MXNet library at runtime
 */

#pragma once

#include <string>
#include <vector>
#include <array>
#include <memory>
#include "dmlc/base.h"
#include "mshadow/base.h"
#include "c_api.h"

/*!
 *\brief whether to use opencv support
 */
#ifndef MXNET_USE_OPENCV
#define MXNET_USE_OPENCV 0
#endif

/*!
 *\brief whether to use cuda support
 */
#ifndef MXNET_USE_CUDA
#define MXNET_USE_CUDA MSHADOW_USE_CUDA
#endif

/*!
 *\brief whether to use cudnn library for convolution
 */
#ifndef MXNET_USE_CUDNN
#define MXNET_USE_CUDNN MSHADOW_USE_CUDNN
#endif

#ifndef MXNET_USE_CUTENSOR
#define MXNET_USE_CUTENSOR MSHADOW_USE_CUTENSOR
#endif

#ifndef MXNET_USE_NVML
#define MXNET_USE_NVML 0
#endif

#ifndef MXNET_USE_NCCL
#define MXNET_USE_NCCL 0
#endif

/*!
 *\brief whether to use cusolver library
 */
#ifndef MXNET_USE_CUSOLVER
#define MXNET_USE_CUSOLVER MSHADOW_USE_CUSOLVER
#endif

/*! \brief Error message for using gpu when MXNET_USE_CUDA==0 */
#define MXNET_GPU_NOT_ENABLED_ERROR "GPU is not enabled"

#ifndef MXNET_USE_TENSORRT
#define MXNET_USE_TENSORRT 0
#endif

#ifndef MXNET_USE_BLAS_ATLAS
#define MXNET_USE_BLAS_ATLAS 0
#endif

#ifndef MXNET_USE_BLAS_OPEN
#define MXNET_USE_BLAS_OPEN 0
#endif

#ifndef MXNET_USE_BLAS_MKL
#define MXNET_USE_BLAS_MKL 0
#endif

#ifndef MXNET_USE_BLAS_APPLE
#define MXNET_USE_BLAS_APPLE 0
#endif

#ifndef MXNET_USE_LAPACK
#define MXNET_USE_LAPACK 0
#endif

#ifndef MXNET_USE_ONEDNN
#define MXNET_USE_ONEDNN 0
#endif

#ifndef MXNET_USE_OPENMP
#define MXNET_USE_OPENMP 0
#endif

#ifndef MXNET_USE_F16C
#define MXNET_USE_F16C MSHADOW_USE_F16C
#endif

#ifndef MXNET_USE_DIST_KVSTORE
#define MXNET_USE_DIST_KVSTORE 0
#endif

#ifndef MXNET_USE_SIGNAL_HANDLER
#define MXNET_USE_SIGNAL_HANDLER 0
#endif

#ifndef MXNET_USE_INT64_TENSOR_SIZE
#define MXNET_USE_INT64_TENSOR_SIZE MSHADOW_INT64_TENSOR_SIZE
#endif

#ifndef MXNET_USE_TVM_OP
#define MXNET_USE_TVM_OP 0
#endif

namespace mxnet {
namespace features {
// Check compile flags such as CMakeLists.txt

/// Compile time features
// ATTENTION: When changing this enum, match the strings in the implementation file!
enum : unsigned {
  // NVIDIA, CUDA
  CUDA = 0,
  CUDNN,
  NCCL,
  TENSORRT,
  CUTENSOR,

  // CPU Features / optimizations
  CPU_SSE,
  CPU_SSE2,
  CPU_SSE3,
  CPU_SSE4_1,
  CPU_SSE4_2,
  CPU_SSE4A,  // AMD extensions to SSE4
  CPU_AVX,
  CPU_AVX2,

  // Multiprocessing / CPU / System
  OPENMP,
  SSE,
  F16C,
  JEMALLOC,

  // Math libraries & BLAS
  // Flavour of BLAS
  BLAS_OPEN,
  BLAS_ATLAS,
  // Intel(R) Math Kernel Library
  BLAS_MKL,
  BLAS_APPLE,
  // Other math libraries:
  // Linear Algebra PACKage
  LAPACK,
  // oneAPI Deep Neural Network Library (oneDNN)
  ONEDNN,

  // Image processing
  OPENCV,

  // Misc
  DIST_KVSTORE,
  INT64_TENSOR_SIZE,

  // Signal handler to print stack traces on exceptions
  SIGNAL_HANDLER,
  DEBUG,

  // TVM operator
  TVM_OP,

  // size indicator
  MAX_FEATURES
};

struct EnumNames {
  static const std::vector<std::string> names;
};

struct LibInfo {
  LibInfo();
  static LibInfo* getInstance();
  const std::array<LibFeature, MAX_FEATURES>& getFeatures() {
    return m_lib_features;
  }

 private:
  std::array<LibFeature, MAX_FEATURES> m_lib_features;
  static std::unique_ptr<LibInfo> m_inst;
};

/*!
 * \return true if the given feature is supported
 */
bool is_enabled(unsigned feat);

}  // namespace features
}  // namespace mxnet
