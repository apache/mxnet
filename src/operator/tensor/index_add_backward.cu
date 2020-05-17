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

template<typename xpu, typename DType>
void IndexAddOpBackwardACalc(mshadow::Stream<xpu> *s,
                             DType* grad_a, const DType* ograd,
                             const size_t* stride,
                             const size_t tail_size, const int ind_num,
                             const int ind_ndim, const int* ind_vec,
                             const int req, const int out_ndim) {
  using namespace mxnet_op;
  using namespace mshadow;
  int* d_ind_vec;
  cudaMalloc(reinterpret_cast<void**>(&d_ind_vec), sizeof(int) * ind_ndim * ind_num);
  cudaMemcpy(d_ind_vec, ind_vec, sizeof(int) * ind_ndim * ind_num, cudaMemcpyHostToDevice);
  size_t* d_stride;
  size_t shape_size = sizeof(size_t) * out_ndim;
  cudaMalloc(reinterpret_cast<void**>(&d_stride), shape_size);
  cudaMemcpy(d_stride, stride, shape_size, cudaMemcpyHostToDevice);
  Kernel<IndexAddBackwardAKernel<DType>, xpu>::Launch(
                                             s, ind_num, grad_a, ograd,
                                             d_stride, tail_size,
                                             ind_num, ind_ndim, d_ind_vec, req);
  cudaFree(d_ind_vec);
  cudaFree(d_stride);
}

template<typename DType>
struct IndexAddBackwardValGPUKernel {
  MSHADOW_XINLINE static void Map(size_t i, DType* grad_val,
                                  const DType* ograd,
                                  const size_t* ograd_tail_shape,
                                  const size_t* ograd_pre_stride,
                                  const size_t* val_stride,
                                  const size_t* val_shape,
                                  const size_t ograd_tail_size, const int ind_num,
                                  const int ind_ndim, const int* ind_vec,
                                  const int out_ndim) {
    size_t id = 0;
    for (int dim = 0; dim < ind_ndim; ++dim) {
      id += ograd_pre_stride[dim] * ind_vec[dim * ind_num + i];
    }
    id *= ograd_tail_size;
    for (size_t _i = 0; _i < ograd_tail_size; ++_i) {
      size_t ograd_tail_id[MXNET_SPECIAL_MAX_NDIM];
      index_unravel(_i, out_ndim, ograd_tail_shape, ograd_tail_id);
      size_t val_id[MXNET_SPECIAL_MAX_NDIM];
      for (int _j = 0; _j < out_ndim; ++_j) {
        val_id[_j] = (val_shape[_j] == 1) ? 0 : ograd_tail_id[_j];
      }
      val_id[ind_ndim - 1] = (val_shape[ind_ndim - 1] == 1) ? 0 : i;
      size_t val_dest = index_dot(out_ndim, val_id, val_stride);
      atomicAdd(&grad_val[val_dest], ograd[id + _i]);
    }
  }
};

template<typename xpu, typename DType>
void IndexAddOpBackwardValCalc(mshadow::Stream<xpu> *s,
                               DType* grad_val, const DType* ograd,
                               const size_t* ograd_tail_shape,
                               const size_t* ograd_pre_stride,
                               const size_t* val_stride,
                               const size_t* val_shape,
                               const size_t tail_size, const int ind_num,
                               const int ind_ndim, const int* ind_vec,
                               const int out_ndim) {
  using namespace mxnet_op;
  using namespace mshadow;
  int * d_ind_vec;
  size_t *d_ograd_tail_shape, *d_ograd_pre_stride, *d_val_stride, *d_val_shape;
  cudaMalloc(reinterpret_cast<void**>(&d_ind_vec), sizeof(int) * ind_ndim * ind_num);
  cudaMemcpy(d_ind_vec, ind_vec, sizeof(int) * ind_ndim * ind_num, cudaMemcpyHostToDevice);
  size_t shape_size = sizeof(size_t) * out_ndim * 4;
  cudaMalloc(reinterpret_cast<void**>(&d_ograd_tail_shape), shape_size);
  cudaMemcpy(d_ograd_tail_shape, ograd_tail_shape, shape_size, cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void**>(&d_ograd_pre_stride), shape_size);
  cudaMemcpy(d_ograd_pre_stride, ograd_pre_stride, shape_size, cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void**>(&d_val_stride), shape_size);
  cudaMemcpy(d_val_stride, val_stride, shape_size, cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void**>(&d_val_shape), shape_size);
  cudaMemcpy(d_val_shape, val_shape, shape_size, cudaMemcpyHostToDevice);
  Kernel<IndexAddBackwardValGPUKernel<DType>, xpu>::Launch(
    s, ind_num, grad_val, ograd, d_ograd_tail_shape, d_ograd_pre_stride,
    d_val_stride, d_val_shape, tail_size, ind_num, ind_ndim, d_ind_vec, out_ndim);
  cudaFree(d_ind_vec);
  cudaFree(d_ograd_tail_shape);
  cudaFree(d_ograd_pre_stride);
  cudaFree(d_val_stride);
  cudaFree(d_val_shape);
}

NNVM_REGISTER_OP(_backward_index_add)
.set_attr<FCompute>("FCompute<gpu>", IndexAddOpBackward<gpu>);

}  // namespace op
}  // namespace mxnet

