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
 * \file transpose_op-inl.cuh
 * \brief Function definition for cuda transpose
 * \author Chaitanya Bapat
 */

#ifndef MXNET_OPERATOR_TENSOR_TRANSPOSE_OP_INL_CUH_
#define MXNET_OPERATOR_TENSOR_TRANSPOSE_OP_INL_CUH_

#include <mxnet/tuple.h>
#include <mxnet/tensor_blob.h>
#include <mshadow/base.h>
#include <algorithm>
#include <utility>
#include "../../common/cuda_utils.h"

namespace mxnet {
namespace op {
namespace mshadow {
namespace cuda {
template<typename DType>
__global__ void Transpose2DKernel(const DType *in, DType *out, index_t row, index_t col) {
  const index_t TILE_DIM = 32;
  const index_t BLOCK_ROWS = 8;
  __shared__ DType tile[TILE_DIM][TILE_DIM + 1];

  index_t x = blockIdx.x * TILE_DIM + threadIdx.x;
  index_t y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (index_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
      tile[threadIdx.y+j][threadIdx.x] = in[(y+j)*col + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (index_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
      out[(y+j)*row+ x] = tile[threadIdx.x][threadIdx.y + j];
}
}  // namespace cuda
}  // namespace mshadow

template<typename DType, typenamme xpu>
inline typename std::enable_if<std::is_same<xpu, gpu>::value, void>::type
Transpose2D(const DType *in, DType *out, index_t row, index_t col) {
  using namespace mshadow::cuda;
  dim3 grid(32);
  dim3 block(8);
  Transpose2DKernel<DType><<<grid, block>>>(in, out, row, col);
  MSHADOW_CUDA_POST_KERNEL_CHECK(Transpose2DKernel);
}

}  // namespace op
}  // namespace mxnet


#endif  // MXNET_OPERATOR_TENSOR_TRANSPOSE_OP_INL_CUH_
