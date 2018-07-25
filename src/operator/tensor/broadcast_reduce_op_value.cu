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
 *  Copyright (c) 2016 by Contributors
 * \file broadcast_reduce_op.cu
 * \brief GPU Implementation of broadcast and reduce functions.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {

template<>
void L2NormComputeEx<gpu>(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const NormParam& param = nnvm::get<NormParam>(attrs.parsed);
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  const NDArrayStorageType istype = inputs[0].storage_type();
  const TShape axis = param.axis.has_value() ? param.axis.value() : TShape();
  if ((istype == kRowSparseStorage || istype == kCSRStorage) && axis.ndim() == 0 &&
       param.ord == 2) {
    // l2 norm on the entire array
    L2NormComputeSparseImpl<gpu>(s, inputs[0], req[0], outputs[0].data());
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

NNVM_REGISTER_OP(sum)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow::red::sum>);

NNVM_REGISTER_OP(_backward_sum)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseNone<gpu>);

NNVM_REGISTER_OP(mean)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow::red::sum, true>);

NNVM_REGISTER_OP(_backward_mean)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseNone<gpu, true>);

NNVM_REGISTER_OP(prod)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow_op::product>);

NNVM_REGISTER_OP(_backward_prod)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseInOut<gpu, mshadow_op::rdiv>);

NNVM_REGISTER_OP(nansum)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow_op::nansum>);

NNVM_REGISTER_OP(_backward_nansum)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseInOut<gpu, mshadow_op::nansum_grad>);

NNVM_REGISTER_OP(nanprod)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow_op::nanprod>);

NNVM_REGISTER_OP(_backward_nanprod)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseInOut<gpu, mshadow_op::nanprod_grad>);

NNVM_REGISTER_OP(max)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow::red::maximum>);

NNVM_REGISTER_OP(_backward_max)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseInOut<gpu, mshadow_op::eq>);

NNVM_REGISTER_OP(min)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow::red::minimum>);

NNVM_REGISTER_OP(_backward_min)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseInOut<gpu, mshadow_op::eq>);

NNVM_REGISTER_OP(broadcast_axis)
.set_attr<FCompute>("FCompute<gpu>", BroadcastCompute<gpu>);

NNVM_REGISTER_OP(broadcast_to)
.set_attr<FCompute>("FCompute<gpu>", BroadcastCompute<gpu>);

NNVM_REGISTER_OP(broadcast_like)
.set_attr<FCompute>("FCompute<gpu>", BroadcastCompute<gpu>);

NNVM_REGISTER_OP(_broadcast_backward)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow::red::sum>);

NNVM_REGISTER_OP(norm)
.set_attr<FCompute>("FCompute<gpu>", LpNormCompute<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", L2NormComputeEx<gpu>);

NNVM_REGISTER_OP(_backward_norm)
.set_attr<FCompute>("FCompute<gpu>", LpNormGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet
