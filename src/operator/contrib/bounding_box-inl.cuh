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
 * \file bounding_box-inl.cuh
 * \brief bounding box CUDA operators
 * \author Joshua Zhang
*/
#ifndef MXNET_OPERATOR_CONTRIB_BOUNDING_BOX_INL_CUH_
#define MXNET_OPERATOR_CONTRIB_BOUNDING_BOX_INL_CUH_
#include <mxnet/operator_util.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {

template<typename DType>
struct valid_score {
  DType thresh;
  explicit valid_score(DType _thresh) : thresh(_thresh) {}
  __host__ __device__ bool operator()(const DType x) {
    return x > thresh;
  }
};

template<typename DType>
int FilterScores(mshadow::Tensor<gpu, 1, DType> out_scores,
                 mshadow::Tensor<gpu, 1, int32_t> out_sorted_index,
                 mshadow::Tensor<gpu, 1, DType> scores,
                 mshadow::Tensor<gpu, 1, int32_t> sorted_index,
                 float valid_thresh) {
  valid_score<DType> pred(static_cast<DType>(valid_thresh));
  DType * end_scores = thrust::copy_if(thrust::device, scores.dptr_, scores.dptr_ + scores.MSize(),
                                       out_scores.dptr_, pred);
  thrust::copy_if(thrust::device, sorted_index.dptr_, sorted_index.dptr_ + sorted_index.MSize(),
                  scores.dptr_, out_sorted_index.dptr_, pred);
  return end_scores - out_scores.dptr_;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_BOUNDING_BOX_INL_CUH_
