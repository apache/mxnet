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
 * \file square_sum.cc
 * \brief CPU Implementation of square_sum op.
 */
#include "./square_sum-inl.h"

namespace mxnet {
namespace op {

template<>
void CheckSameIdx<cpu>(const OpContext& ctx,
                       const TBlob& ograd_row_idx,
                       const TBlob& in_row_idx) {
  MSHADOW_IDX_TYPE_SWITCH(ograd_row_idx.type_flag_, IType, {
    mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
    const IType* ograd_idx = ograd_row_idx.dptr<IType>();
    const IType* in_idx = in_row_idx.dptr<IType>();
    const nnvm::dim_t idx_size = ograd_row_idx.Size();
    int32_t is_different = 0;
    mxnet_op::Kernel<CheckSameIdxKernel, cpu>::Launch(s, idx_size,
      ograd_idx, in_idx, &is_different);
    CHECK_EQ(is_different, 0) << "SquareSumRspGradImpl only supports"
                                 " equal ograd_row_idx and input_row_idx"
                                 " when ograd and input are both"
                                 " row-sparse and input data is not a full"
                                 " row-sparse matrix";
  })
}


MXNET_OPERATOR_REGISTER_REDUCE(_square_sum)
.describe(R"code(Computes the square sum of array elements over a given axis
for row-sparse matrix. This is a temporary solution for fusing ops square and
sum together for row-sparse matrix to save memory for storing gradients.
It will become deprecated once the functionality of fusing operators is finished
in the future.

Example::

  dns = mx.nd.array([[0, 0], [1, 2], [0, 0], [3, 4], [0, 0]])
  rsp = dns.tostype('row_sparse')
  sum = mx.nd._internal._square_sum(rsp, axis=1)
  sum = [0, 5, 0, 25, 0]
)code" ADD_FILELINE)
.set_attr<FInferStorageType>("FInferStorageType", SquareSumForwardInferStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SquareSumOpForwardEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_square_sum"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_square_sum)
.set_num_inputs(2)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FInferStorageType>("FInferStorageType", SquareSumBackwardInferStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SquareSumOpBackwardEx<cpu>);

}  // namespace op
}  // namespace mxnet
