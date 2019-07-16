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
 * Copyright (c) 2019 by Contributors
 * \file cuda_utils.cc
 * \brief CUDA debugging utilities.
 */

#include <mxnet/base.h>
#include "cuda_utils.h"

#if MXNET_USE_CUDA == 1

namespace mxnet {
namespace common {
namespace cuda {

// The oldest version of cuda used in upstream MXNet CI testing, both for unix and windows.
// Users that have rebuilt MXNet against older versions will we advised with a warning to upgrade
// their systems to match the CI level.  Minimally, users should rerun the CI locally.
#if defined(_MSC_VER)
#define MXNET_CI_OLDEST_CUDA_VERSION  9020
#else
#define MXNET_CI_OLDEST_CUDA_VERSION 10000
#endif

// Dynamic init here will emit a warning if runtime and compile-time cuda lib versions mismatch.
// Also if the user has recompiled their source to a version no longer tested by upstream CI.
bool cuda_version_check_performed = []() {
  // MXNet might be built on a machine with a cuda toolkit, but no GPUs or GPU driver.
  // To allow that machine to execute say: python -c 'import mxnet; print(mxnet.__version__)',
  // we won't perform a check if there is no driver.  Any actual attempt to use the cuda API's
  // will yield the desired message: CUDA driver version is insufficient for CUDA runtime version.
  int cuda_driver_version = 0;
  CUDA_CALL(cudaDriverGetVersion(&cuda_driver_version));
  // Also, don't bother with checks if there are no GPUs visible (e.g. with CUDA_VISIBLE_DEVICES="")
  if (dmlc::GetEnv("MXNET_CUDA_VERSION_CHECKING", true) && cuda_driver_version > 0
                                                        && Context::GetGPUCount() > 0) {
    // Not currently performing a runtime check of linked-against vs. compiled-against
    // cuda runtime library, as major.minor must match for libmxnet.so to even load, per:
    // https://docs.nvidia.com/deploy/cuda-compatibility/#binary-compatibility
    if (CUDA_VERSION < MXNET_CI_OLDEST_CUDA_VERSION)
      LOG(WARNING) << "Upgrade advisory: this mxnet has been built against cuda library version "
                   << CUDA_VERSION << ", which is older than the oldest version tested by CI ("
                   << MXNET_CI_OLDEST_CUDA_VERSION << ").  "
                   << "Set MXNET_CUDA_VERSION_CHECKING=0 to quiet this warning.";
  }
  return true;
}();

}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#if MXNET_USE_CUDNN == 1

namespace mxnet {
namespace common {
namespace cudnn {

// The oldest version of CUDNN used in upstream MXNet CI testing, both for unix and windows.
// Users that have rebuilt MXNet against older versions will we advised with a warning to upgrade
// their systems to match the CI level.  Minimally, users should rerun the CI locally.
#if defined(_MSC_VER)
#define MXNET_CI_OLDEST_CUDNN_VERSION 7600
#else
#define MXNET_CI_OLDEST_CUDNN_VERSION 7600
#endif

// Dynamic init here will emit a warning if runtime and compile-time cudnn lib versions mismatch.
// Also if the user has recompiled their source to a version no longer tested by upstream CI.
bool cudnn_version_check_performed = []() {
  // MXNet might be built on a machine with a cuda toolkit, but no GPUs or GPU driver.
  // To allow that machine to execute say: python -c 'import mxnet; print(mxnet.__version__)',
  // we won't perform a check if there is no driver.  Any actual attempt to use the cuda API's
  // will yield the desired message: CUDA driver version is insufficient for CUDA runtime version.
  int cuda_driver_version = 0;
  CUDA_CALL(cudaDriverGetVersion(&cuda_driver_version));
  // Also, don't bother with checks if there are no GPUs visible (e.g. with CUDA_VISIBLE_DEVICES="")
  if (dmlc::GetEnv("MXNET_CUDNN_VERSION_CHECKING", true) && cuda_driver_version > 0
                                                         && Context::GetGPUCount() > 0) {
    size_t linkedAgainstCudnnVersion = cudnnGetVersion();
    if (linkedAgainstCudnnVersion != CUDNN_VERSION)
      LOG(WARNING) << "cuDNN library mismatch: linked-against version " << linkedAgainstCudnnVersion
                   << " != compiled-against version " << CUDNN_VERSION << ".  "
                   << "Set MXNET_CUDNN_VERSION_CHECKING=0 to quiet this warning.";
    if (CUDNN_VERSION < MXNET_CI_OLDEST_CUDNN_VERSION)
      LOG(WARNING) << "Upgrade advisory: this mxnet has been built against cuDNN library version "
                   <<  CUDNN_VERSION << ", which is older than the oldest version tested by CI ("
                   << MXNET_CI_OLDEST_CUDNN_VERSION << ").  "
                   << "Set MXNET_CUDNN_VERSION_CHECKING=0 to quiet this warning.";
  }
  return true;
}();

}  // namespace cudnn
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDNN
