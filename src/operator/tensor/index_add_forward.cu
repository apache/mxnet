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
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_tail_shape,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_pre_stride,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_stride,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_shape,
                                  const int a_tail_size, const int ind_num,
                                  const int ind_ndim, const int* ind,
                                  const int a_ndim) {
    index_t id = 0;
    int seg = MXNET_SPECIAL_MAX_NDIM - a_ndim;
    for (int dim = 0; dim < ind_ndim; ++dim) {
      id += a_pre_stride[seg + dim] * ind[dim * ind_num + i];
    }
    id *= a_tail_size;
    for (int _i = 0; _i < a_tail_size; ++_i) {
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_tail_id = mxnet_op::unravel(_i, a_tail_shape);
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_id;
      for (int _j = seg; _j < seg + a_ndim; ++_j) {
        val_id[_j] = (val_shape[_j] == 1) ? 0 : a_tail_id[_j];
      }
      val_id[seg + ind_ndim - 1] = (val_shape[seg + ind_ndim - 1] == 1) ? 0 : i;
      index_t val_dest = mxnet_op::dot(val_id, val_stride);
      atomicAdd(&out[id + _i], val[val_dest]);
    }
  }
};

template<typename xpu, typename DType>
void IndexAddForwardCalc(mshadow::Stream<xpu> *s,
                         const int ind_num, DType* out,
                         const DType* val,
                         const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_tail_shape,
                         const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_pre_stride,
                         const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_stride,
                         const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_shape,
                         const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_shape,
                         const int a_tail_size,
                         const int ind_ndim, const int* ind,
                         const int a_ndim) {
  using namespace mxnet_op;
  using namespace mshadow;
  Kernel<IndexAddForwardGPUKernel<DType>, xpu>::Launch(
                                              s, ind_num, out, val,
                                              a_tail_shape, a_pre_stride,
                                              val_stride, val_shape,
                                              a_tail_size, ind_num,
                                              ind_ndim, ind, a_ndim);
}


NNVM_REGISTER_OP(_npx_index_add)
.set_attr<FCompute>("FCompute<gpu>", IndexAddOpForward<gpu>);

}  // namespace op
}  // namespace mxnet

