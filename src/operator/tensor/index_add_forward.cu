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

template<typename DType, typename VType, int NDim>
struct IndexAddForwardGPUKernel {
  MSHADOW_XINLINE static void Map(size_t i, DType* out,
                                  const VType* val,
                                  const mshadow::Shape<NDim> a_tail_shape,
                                  const mshadow::Shape<NDim> a_pre_stride,
                                  const mshadow::Shape<NDim> val_stride,
                                  const mshadow::Shape<NDim> val_shape,
                                  const size_t a_tail_size, const int ind_num,
                                  const int ind_ndim, const int* ind_vec) {
    size_t id = 0;
    for (int dim = 0; dim < ind_ndim; ++dim) {
      id += a_pre_stride[dim] * ind_vec[dim * ind_num + i];
    }
    id *= a_tail_size;
    for (int _i = 0; _i < a_tail_size; ++_i) {
      mshadow::Shape<NDim> a_tail_id = mxnet_op::unravel(_i, a_tail_shape);
      mshadow::Shape<NDim> val_id;
      for (int _j = 0; _j < NDim; ++_j) {
        val_id[_j] = (val_shape[_j] == 1) ? 0 : a_tail_id[_j];
      }
      val_id[ind_ndim - 1] = (val_shape[ind_ndim - 1] == 1) ? 0 : i;
      size_t val_dest = mxnet_op::dot(val_id, val_stride);
      atomicAdd(&out[id + _i], static_cast<DType>(val[val_dest]));
    }
  }
};

template<typename xpu, typename DType, typename VType, int NDim>
void IndexAddForwardCalc(mshadow::Stream<xpu> *s,
                         const int ind_num, DType* out,
                         const VType* val,
                         const mshadow::Shape<NDim>& a_tail_shape,
                         const mshadow::Shape<NDim>& a_pre_stride,
                         const mshadow::Shape<NDim>& val_stride,
                         const mshadow::Shape<NDim>& val_shape,
                         const size_t a_tail_size,
                         const int ind_ndim, const int* ind_vec) {
  using namespace mxnet_op;
  using namespace mshadow;
  int* d_ind_vec;
  cudaMalloc(reinterpret_cast<void**>(&d_ind_vec), sizeof(int) * ind_ndim * ind_num);
  cudaMemcpy(d_ind_vec, ind_vec, sizeof(int) * ind_ndim * ind_num, cudaMemcpyHostToDevice);
  Kernel<IndexAddForwardGPUKernel<DType, VType, NDim>, xpu>::Launch(
                                              s, ind_num, out, val,
                                              a_tail_shape, a_pre_stride,
                                              val_stride, val_shape,
                                              a_tail_size, ind_num,
                                              ind_ndim, d_ind_vec);
}


NNVM_REGISTER_OP(_npx_index_add)
.set_attr<FCompute>("FCompute<gpu>", IndexAddOpForward<gpu>);

}  // namespace op
}  // namespace mxnet

