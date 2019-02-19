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
 *  Copyright (c) 2018 by Contributors
 * \file libinfo.cc
 * \author larroy
 * \brief check MXNet features including compile time support
 */

#include "mxnet/libinfo.h"
#include <bitset>
#include "mxnet/base.h"

namespace mxnet {
namespace features {

class FeatureSet {
 public:
  FeatureSet() :
      feature_bits() {
    // GPU
    feature_bits.set(CUDA, MXNET_USE_CUDA);
    feature_bits.set(CUDNN, MXNET_USE_CUDNN);
    feature_bits.set(NCCL, MXNET_USE_NCCL);
    feature_bits.set(CUDA_RTC, MXNET_ENABLE_CUDA_RTC);
    feature_bits.set(TENSORRT, MXNET_USE_TENSORRT);

    // Check flags for example with gcc -msse3 -mavx2 -dM -E - < /dev/null | egrep "SSE|AVX"
#if __SSE__
    feature_bits.set(CPU_SSE);
#endif
#if __SSE2__
    feature_bits.set(CPU_SSE2);
#endif
#if __SSE3__
    feature_bits.set(CPU_SSE3);
#endif
#if __SSE4_1__
    feature_bits.set(CPU_SSE4_1);
#endif
#if __SSE4_2__
    feature_bits.set(CPU_SSE4_2);
#endif
#if __SSE4A__
    feature_bits.set(CPU_SSE4A);
#endif
#if __AVX__
    feature_bits.set(CPU_AVX);
#endif
#if __AVX2__
    feature_bits.set(CPU_AVX2);
#endif

    // CPU
    feature_bits.set(OPENMP, MXNET_USE_OPENMP);
    feature_bits.set(F16C, MXNET_USE_F16C);

    // Math
    feature_bits.set(BLAS_OPEN, MXNET_USE_BLAS_OPEN);
    feature_bits.set(BLAS_ATLAS, MXNET_USE_BLAS_ATLAS);
    feature_bits.set(BLAS_MKL, MXNET_USE_BLAS_MKL);
    feature_bits.set(BLAS_APPLE, MXNET_USE_BLAS_APPLE);
    feature_bits.set(LAPACK, MXNET_USE_LAPACK);
    feature_bits.set(MKLDNN, MXNET_USE_MKLDNN);

    // Image
    feature_bits.set(OPENCV, MXNET_USE_OPENCV);

    // Misc
    feature_bits.set(CAFFE, MXNET_USE_CAFFE);
    feature_bits.set(DIST_KVSTORE, MXNET_USE_DIST_KVSTORE);
    feature_bits.set(SIGNAL_HANDLER, MXNET_USE_SIGNAL_HANDLER);
#ifndef NDEBUG
    feature_bits.set(DEBUG);
#endif

#if USE_JEMALLOC == 1
    feature_bits.set(JEMALLOC);
#endif
  }
  bool is_enabled(const unsigned feat) const {
    CHECK_LT(feat, MAX_FEATURES);
    return feature_bits.test(feat);
  }

 private:
  std::bitset<MAX_FEATURES> feature_bits;
};

static FeatureSet featureSet;

bool is_enabled(const unsigned feat) {
  return featureSet.is_enabled(feat);
}

LibInfo::LibInfo() {
    for (size_t i = 0; i < MAX_FEATURES; ++i) {
        m_lib_features[i].name = EnumNames::names[i].c_str();
        m_lib_features[i].enabled = is_enabled(i);
    }
}

LibInfo *LibInfo::getInstance() {
    if (!m_inst)
        m_inst = std::make_unique<LibInfo>();
    return m_inst.get();
}

std::unique_ptr<LibInfo> LibInfo::m_inst = nullptr;

const std::vector<std::string> EnumNames::names = {
  "CUDA",
  "CUDNN",
  "NCCL",
  "CUDA_RTC",
  "TENSORRT",
  "CPU_SSE",
  "CPU_SSE2",
  "CPU_SSE3",
  "CPU_SSE4_1",
  "CPU_SSE4_2",
  "CPU_SSE4A",
  "CPU_AVX",
  "CPU_AVX2",
  "OPENMP",
  "SSE",
  "F16C",
  "JEMALLOC",
  "BLAS_OPEN",
  "BLAS_ATLAS",
  "BLAS_MKL",
  "BLAS_APPLE",
  "LAPACK",
  "MKLDNN",
  "OPENCV",
  "CAFFE",
  "PROFILER",
  "DIST_KVSTORE",
  "CXX14",
  "SIGNAL_HANDLER",
  "DEBUG",
};

}  // namespace features
}  // namespace mxnet
