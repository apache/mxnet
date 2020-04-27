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

template<typename xpu, typename DType, typename VType, int NDim>
void IndexAddForwardImpl(mshadow::Stream<xpu> *s,
                          const int ind_num, DType* out,
                        const VType* val,
                        const mshadow::Shape<NDim>& a_tail_shape,
                        const mshadow::Shape<NDim>& a_pre_stride,
                        const mshadow::Shape<NDim>& val_stride,
                        const mshadow::Shape<NDim>& val_shape,
                        const size_t a_tail_size,
                        const int ind_ndim, const int* ind_vec,
                        const int req) {
  using namespace mxnet_op;
  using namespace mshadow;
  int * d_ind_vec;
  cudaMalloc(reinterpret_cast<void**>(&d_ind_vec), sizeof(int) * ind_ndim * ind_num);
  cudaMemcpy(d_ind_vec, ind_vec, sizeof(int) * ind_ndim * ind_num, cudaMemcpyHostToDevice);
  Kernel<IndexAddForwardKernel<DType, VType, NDim>, xpu>::Launch(
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

