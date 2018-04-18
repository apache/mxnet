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
 * \file sequence_op_common.h
 * \brief common function used for sequence layers
 * \author Sebastian Bodenstein
*/
#ifndef MXNET_OPERATOR_SEQUENCE_OP_COMMON_H_
#define MXNET_OPERATOR_SEQUENCE_OP_COMMON_H_
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <vector>
#include "./operator_common.h"

namespace mxnet {
namespace op {

template <typename DType, typename RType>
typename std::enable_if<std::is_integral<RType>::value>::type
inline IndexTensorToVector(mshadow::Tensor<gpu, 1, DType> data,
                           std::vector<RType> *index_vec) {
#if MXNET_USE_CUDA
  size_t const max_seq_len = data.shape_.Size();
  DType *temp_index =
      reinterpret_cast<DType *>(malloc(sizeof(DType) * max_seq_len));
  cudaError_t cuda_status =
      cudaMemcpyAsync(temp_index, data.dptr_, max_seq_len * sizeof(DType),
                      cudaMemcpyDeviceToHost, data.stream_->stream_);
  CHECK_EQ(cuda_status, cudaSuccess) << "cuda memcpy label error";
  for (size_t i = 0; i < max_seq_len; ++i) {
    (*index_vec)[i] = static_cast<RType>(std::lround(temp_index[i]));
  }
  free(temp_index);
#endif
}
template <typename DType, typename RType>
typename std::enable_if<std::is_integral<RType>::value>::type
inline IndexTensorToVector(mshadow::Tensor<cpu, 1, DType> data,
                           std::vector<RType> *index_vec) {
  int max_seq_len = data.shape_.Size();
  DType *index_array = static_cast<DType *>(data.dptr_);
  for (int i = 0; i < max_seq_len; ++i)
    (*index_vec)[i] = static_cast<RType>(std::lround(index_array[i]));
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SEQUENCE_OP_COMMON_H_
