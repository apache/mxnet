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
#include "./index_update-inl.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType, typename VType, int NDim>
void IndexUpdateForwardImpl(mshadow::Stream<xpu> *s,
                            const int ind_num, DType* out,
                            const VType* val,
                            const mshadow::Shape<NDim>& a_tail_shape,
                            const mshadow::Shape<NDim>& a_pre_stride,
                            const mshadow::Shape<NDim>& val_stride,
                            const mshadow::Shape<NDim>& val_shape,
                            const size_t a_tail_size,
                            const int ind_ndim, const int* ind_vec,
                            const int req, int64_t* pre) {
  using namespace mxnet_op;
  using namespace mshadow;
  Kernel<IndexUpdateForwardKernel<DType, VType, NDim>, xpu>::Launch(
    s, ind_num, out, val, a_tail_shape, a_pre_stride, val_stride,
    val_shape, a_tail_size, ind_num, ind_ndim, ind_vec, req, pre);
}

// DMLC_REGISTER_PARAMETER(IndexModifyParam);

NNVM_REGISTER_OP(_npx_index_update)
.describe(R"code(Implent a[idx] = val.
Returns the value of a that would result from the NumPy-style indexed assignment:
a[idx] = val
Note x itself is not modified, instead the new value that x would have taken is returned.
If multiple indices refer to the same location it is undefined which update is chosen;
it may choose the order of updates arbitrarily and nondeterministically
(e.g., due to concurrent updates on some hardware platforms).
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<IndexModifyParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "val"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", IndexModifyOpShape)
.set_attr<nnvm::FInferType>("FInferType", IndexModifyOpType)
.set_attr<FCompute>("FCompute<cpu>", IndexUpdateOpForward<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("a", "NDArray-or-Symbol", "Input ndarray")
.add_argument("val", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(IndexModifyParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

