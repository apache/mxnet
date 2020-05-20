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
 * \file index_add-inl.cc
 * \brief CPU implementation of index_add operator
*/
#include <vector>
#include "./index_add-inl.h"

namespace mxnet {
namespace op {
template<typename DType>
struct IndexAddForwardCPUKernel {
  MSHADOW_XINLINE static void Map(size_t i, DType* out,
                                  const DType* val,
                                  const size_t* a_tail_shape,
                                  const size_t* a_pre_stride,
                                  const size_t* val_stride,
                                  const size_t* val_shape,
                                  const size_t* a_shape,
                                  const size_t a_tail_size, const int ind_num,
                                  const int ind_ndim, const int* ind,
                                  const int a_ndim) {
    size_t id = 0;
    for (int dim = 0; dim < ind_ndim; ++dim) {
      CHECK_LT(ind[dim * ind_num + i], a_shape[dim])
        << "IndexError: index " << ind[dim * ind_num + i]
        << " is out of bounds for axis " << dim
        << " with size " << a_shape[dim];
      CHECK_GE(ind[dim * ind_num + i], 0)
        << "IndexError: index " << ind[dim * ind_num + i]
        << " should be greater or equal to 0.";
      id += a_pre_stride[dim] * ind[dim * ind_num + i];
    }
    id *= a_tail_size;
    #pragma omp parallel for
    for (size_t _i = 0; _i < a_tail_size; ++_i) {
      size_t a_tail_id[MXNET_SPECIAL_MAX_NDIM];
      index_unravel(_i, a_ndim, a_tail_shape, a_tail_id);
      size_t val_id[MXNET_SPECIAL_MAX_NDIM];
      for (int _j = 0; _j < a_ndim; ++_j) {
        val_id[_j] = (val_shape[_j] == 1) ? 0 : a_tail_id[_j];
      }
      val_id[ind_ndim - 1] = (val_shape[ind_ndim - 1] == 1) ? 0 : i;
      size_t val_dest = index_dot(a_ndim, val_id, val_stride);
      #pragma omp critical
      {
        out[id + _i] += val[val_dest];
      }
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
  Kernel<IndexAddForwardCPUKernel<DType>, xpu>::Launch(
                                             s, ind_num, out, val,
                                             a_tail_shape, a_pre_stride,
                                             val_stride, val_shape, a_shape,
                                             a_tail_size, ind_num,
                                             ind_ndim, ind, a_ndim);
}



NNVM_REGISTER_OP(_npx_index_add)
.describe(R"code(This operators implements the "+=" mimic function.
)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "ind", "val"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", IndexModifyOpShape)
.set_attr<nnvm::FInferType>("FInferType", IndexModifyOpType)
.set_attr<FCompute>("FCompute<cpu>", IndexAddOpForward<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_index_add"})
.add_argument("a", "NDArray-or-Symbol", "Input ndarray")
.add_argument("ind", "NDArray-or-Symbol", "Index ndarray")
.add_argument("val", "NDArray-or-Symbol", "Input ndarray");
}  // namespace op
}  // namespace mxnet

