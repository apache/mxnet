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
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_tail_shape,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_pre_stride,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_stride,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_shape,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_shape,
                                  const int a_tail_size, const int ind_num,
                                  const int ind_ndim, const int* ind,
                                  const int a_ndim, const int seg) {
    index_t id = 0;
    for (int dim = 0; dim < ind_ndim; ++dim) {
      CHECK_LT(ind[dim * ind_num + i], a_shape[seg + dim])
        << "IndexError: index " << ind[dim * ind_num + i]
        << " is out of bounds for axis " << dim
        << " with size " << a_shape[seg + dim];
      CHECK_GE(ind[dim * ind_num + i], 0)
        << "IndexError: index " << ind[dim * ind_num + i]
        << " should be greater or equal to 0.";
      id += a_pre_stride[seg + dim] * ind[dim * ind_num + i];
    }
    id *= a_tail_size;
    for (int _i = 0; _i < a_tail_size; ++_i) {
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_tail_id = mxnet_op::unravel(_i, a_tail_shape);
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_id;
      for (int _j = 0; _j < seg; ++_j) {
        val_id[_j] = 0;
      }
      for (int _j = seg; _j < seg + a_ndim; ++_j) {
        val_id[_j] = (val_shape[_j] == 1) ? 0 : a_tail_id[_j];
      }
      val_id[seg + ind_ndim - 1] = (val_shape[seg + ind_ndim - 1] == 1) ? 0 : i;
      index_t val_dest = mxnet_op::dot(val_id, val_stride);
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
  int seg = MXNET_SPECIAL_MAX_NDIM - a_ndim;
  Kernel<IndexAddForwardCPUKernel<DType>, xpu>::Launch(
                                             s, ind_num, out, val,
                                             a_tail_shape, a_pre_stride,
                                             val_stride, val_shape, a_shape,
                                             a_tail_size, ind_num,
                                             ind_ndim, ind, a_ndim, seg);
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
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      auto a_grad = MakeNode("_copy", n->attrs.name + "_backward_a",
                              {ograds[0]}, nullptr, &n);
      auto idx_grad = MakeNode("zeros_like", n->attrs.name + "_backward_indices",
                              {n->inputs[1]}, nullptr, &n);
      auto val_grad = MakeNode("_backward_index_add_val", n->attrs.name + "_backward_val",
                              {ograds[0], n->inputs[1]}, nullptr, &n);
      std::vector<nnvm::NodeEntry> ret;
      ret.emplace_back(a_grad);
      ret.emplace_back(idx_grad);
      ret.emplace_back(val_grad);
      return ret;
  })
.add_argument("a", "NDArray-or-Symbol", "Input ndarray")
.add_argument("ind", "NDArray-or-Symbol", "Index ndarray")
.add_argument("val", "NDArray-or-Symbol", "Input ndarray");
}  // namespace op
}  // namespace mxnet

