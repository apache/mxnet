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
 * \file index_add.cu
 * \brief GPU implementation of index_add operator
 */

#include <cub/cub.cuh>
#include "./index_add-inl.h"
#include "../tensor/util/tensor_util-inl.cuh"
#include "../tensor/util/tensor_util-inl.h"

namespace mxnet {
namespace op {

template<typename DType>
struct IndexAddForwardGPUKernel {
  MSHADOW_XINLINE static void Map(size_t i, DType* out,
                                  const DType* val,
                                  const size_t* a_tail_shape,
                                  const size_t* a_pre_stride,
                                  const size_t* val_stride,
                                  const size_t* val_shape,
                                  const size_t a_tail_size, const int ind_num,
                                  const int ind_ndim, const int* ind,
                                  const int a_ndim) {
    size_t id = 0;
    for (int dim = 0; dim < ind_ndim; ++dim) {
      id += a_pre_stride[dim] * ind[dim * ind_num + i];
    }
    id *= a_tail_size;
    for (size_t _i = 0; _i < a_tail_size; ++_i) {
      size_t a_tail_id[MXNET_SPECIAL_MAX_NDIM];
      index_unravel(_i, a_ndim, a_tail_shape, a_tail_id);
      size_t val_id[MXNET_SPECIAL_MAX_NDIM];
      for (int _j = 0; _j < a_ndim; ++_j) {
        val_id[_j] = (val_shape[_j] == 1) ? 0 : a_tail_id[_j];
      }
      val_id[ind_ndim - 1] = (val_shape[ind_ndim - 1] == 1) ? 0 : i;
      size_t val_dest = index_dot(a_ndim, val_id, val_stride);
      atomicAdd(&out[id + _i], val[val_dest]);
    }
  }
};

template<typename xpu, typename DType>
void IndexAddForwardCalc(mshadow::Stream<xpu> *s,
                         const int ind_num, DType* out,
                         const DType* val,
                         const size_t* a_tail_shape,
                         const size_t* a_pre_stride,
                         const size_t* val_stride,
                         const size_t* val_shape,
                         const size_t* a_shape,
                         const size_t a_tail_size,
                         const int ind_ndim, const int* ind,
                         const int a_ndim) {
  using namespace mxnet_op;
  using namespace mshadow;
  size_t *d_a_tail_shape, *d_a_pre_stride, *d_val_stride, *d_val_shape;
  size_t shape_size = sizeof(size_t) * a_ndim;
  cudaMalloc(reinterpret_cast<void**>(&d_a_tail_shape), shape_size);
  cudaMemcpy(d_a_tail_shape, a_tail_shape, shape_size, cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void**>(&d_a_pre_stride), shape_size);
  cudaMemcpy(d_a_pre_stride, a_pre_stride, shape_size, cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void**>(&d_val_stride), shape_size);
  cudaMemcpy(d_val_stride, val_stride, shape_size, cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void**>(&d_val_shape), shape_size);
  cudaMemcpy(d_val_shape, val_shape, shape_size, cudaMemcpyHostToDevice);
  Kernel<IndexAddForwardGPUKernel<DType>, xpu>::Launch(
                                              s, ind_num, out, val,
                                              d_a_tail_shape, d_a_pre_stride,
                                              d_val_stride, d_val_shape,
                                              a_tail_size, ind_num,
                                              ind_ndim, ind, a_ndim);
  cudaFree(d_a_tail_shape);
  cudaFree(d_a_pre_stride);
  cudaFree(d_val_stride);
  cudaFree(d_val_shape);
}


NNVM_REGISTER_OP(_npx_index_add)
.set_attr<FCompute>("FCompute<gpu>", IndexAddOpForward<gpu>);

}  // namespace op
}  // namespace mxnet

