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
 * \file mxfeatures.cc
 * \brief check MXNet features including compile time support
 */

#include "mxnet/mxfeatures.h"
#include "dmlc/logging.h"
#include <bitset>


namespace mxnet {
namespace features {

class Storage {
public:
    Storage():
        feature_bits()
    {
        if (MXNET_USE_CUDA)
            feature_bits.set(CUDA);
        if (MXNET_USE_CUDNN)
            feature_bits.set(CUDNN);
        if (MXNET_USE_NCCL)
            feature_bits.set(NCCL);
        if (MXNET_USE_OPENCV)
            feature_bits.set(OPENCV);
        if (MXNET_ENABLE_CUDA_RTC)
            feature_bits.set(CUDA_RTC);
        if (MXNET_USE_TENSORRT)
            feature_bits.set(TENSORRT);
        if (MXNET_USE_OPENMP)
            feature_bits.set(OPENMP);
        if (MXNET_USE_F16C)
            feature_bits.set(F16C);
        if (MXNET_USE_LAPACK)
            feature_bits.set(LAPACK);
        if (MXNET_USE_MKLDNN)
            feature_bits.set(MKLDNN);
        if (MXNET_USE_OPENCV)
            feature_bits.set(OPENCV);
        if (MXNET_USE_CAFFE)
            feature_bits.set(CAFFE);
        if (MXNET_USE_DIST_KVSTORE)
            feature_bits.set(DIST_KVSTORE);
        if (MXNET_USE_SIGNAL_HANDLER)
            feature_bits.set(SIGNAL_HANDLER);
#ifndef NDEBUG
        feature_bits.set(DEBUG);
#endif


#if USE_JEMALLOC == 1
        feature_bits.set(JEMALLOC);
#endif
    }
    bool is_enabled(unsigned feat) {
        CHECK_LT(feat, MAX_FEATURES);
        return feature_bits.test(feat);
    }
private:
    std::bitset<MAX_FEATURES> feature_bits;
};

static Storage storage;

bool is_enabled(unsigned feat) {
    return storage.is_enabled(feat);
}

}  // namespace features
}  // namespace mxnet
