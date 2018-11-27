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
 *  Copyright (c) 2017 by Contributors
 * \file digitize_op.cuh
 * \brief CUDA implementations for digitize_op.h
 */
#ifndef MXNET_OPERATOR_TENSOR_DIGITIZE_CUH_
#define MXNET_OPERATOR_TENSOR_DIGITIZE_CUH_
#include "./digitize_op.h"

#include <thrust/binary_search.h>
#include <thrust/distance.h>

#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

namespace mxnet {
namespace op {

template<typename DType, typename BType>
struct DigitizeOp::ForwardKernel<gpu, DType, BType> {
  MSHADOW_XINLINE static void Map(int i, const DType *in_data, DType *out_data, BType *bins,
                                  const bool right) {
    const auto data = in_data[i];
    auto elem = right ? thrust::lower_bound(bins.dptr_, bins.dptr_ + bins.size(0), data)
                      : thrust::upper_bound(bins.dptr_, bins.dptr_ + bins.size(0), data);

    out_data[i] = thrust::distance(bins.dptr_, elem);
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_SORT_OP_INL_CUH_
