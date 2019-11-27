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
 *  Copyright (c) 2019 by Contributors
 * \file base.cc
 * \brief Implementation of base declarations, e.g. context
 */
#include <mxnet/base.h>

namespace mxnet {

#define UNUSED(x) (void)(x)

#if MXNET_USE_CUDA == 1
// The oldest version of cuda used in upstream MXNet CI testing, both for unix and windows.
// Users that have rebuilt MXNet against older versions will we advised with a warning to upgrade
// their systems to match the CI level.  Minimally, users should rerun the CI locally.
#if defined(_MSC_VER)
#define MXNET_CI_OLDEST_CUDA_VERSION  9020
#else
#define MXNET_CI_OLDEST_CUDA_VERSION 10000
#endif

void Context::CudaLibChecks() {
  // One-time init here will emit a warning if no gpus or gpu driver is seen.
  // Also if the user has recompiled their source to a version no longer tested by upstream CI.
  static bool cuda_lib_checks_performed = []() {
    if (dmlc::GetEnv("MXNET_CUDA_LIB_CHECKING", true)) {
      if (!GPUDriverPresent())
        LOG(WARNING) << "Please install cuda driver for GPU use.  No cuda driver detected.";
      else if (GetGPUCount() == 0)
        LOG(WARNING) << "GPU context requested, but no GPUs found.";
      else if (CUDA_VERSION < MXNET_CI_OLDEST_CUDA_VERSION)
        LOG(WARNING) << "Upgrade advisory: this mxnet has been built against cuda library version "
                     << CUDA_VERSION << ", which is older than the oldest version tested by CI ("
                     << MXNET_CI_OLDEST_CUDA_VERSION << ").  "
                     << "Set MXNET_CUDA_LIB_CHECKING=0 to quiet this warning.";
    }
    return true;
  }();
  UNUSED(cuda_lib_checks_performed);
}
#endif  // MXNET_USE_CUDA

#if MXNET_USE_CUDNN == 1
// The oldest version of CUDNN used in upstream MXNet CI testing, both for unix and windows.
// Users that have rebuilt MXNet against older versions will we advised with a warning to upgrade
// their systems to match the CI level.  Minimally, users should rerun the CI locally.
#if defined(_MSC_VER)
#define MXNET_CI_OLDEST_CUDNN_VERSION 7600
#else
#define MXNET_CI_OLDEST_CUDNN_VERSION 7600
#endif

void Context::CuDNNLibChecks() {
  // One-time init here will emit a warning if runtime and compile-time cudnn lib versions mismatch.
  // Also if the user has recompiled their source to a version no longer tested by upstream CI.
  static bool cudnn_lib_checks_performed = []() {
    // Don't bother with checks if there are no GPUs visible (e.g. with CUDA_VISIBLE_DEVICES="")
    if (dmlc::GetEnv("MXNET_CUDNN_LIB_CHECKING", true) && GetGPUCount() > 0) {
      size_t linkedAgainstCudnnVersion = cudnnGetVersion();
      if (linkedAgainstCudnnVersion != CUDNN_VERSION)
        LOG(WARNING) << "cuDNN lib mismatch: linked-against version " << linkedAgainstCudnnVersion
                     << " != compiled-against version " << CUDNN_VERSION << ".  "
                     << "Set MXNET_CUDNN_LIB_CHECKING=0 to quiet this warning.";
      if (CUDNN_VERSION < MXNET_CI_OLDEST_CUDNN_VERSION)
        LOG(WARNING) << "Upgrade advisory: this mxnet has been built against cuDNN lib version "
                     <<  CUDNN_VERSION << ", which is older than the oldest version tested by CI ("
                     << MXNET_CI_OLDEST_CUDNN_VERSION << ").  "
                     << "Set MXNET_CUDNN_LIB_CHECKING=0 to quiet this warning.";
    }
    return true;
  }();
  UNUSED(cudnn_lib_checks_performed);
}
#endif  // MXNET_USE_CUDNN

}  // namespace mxnet
