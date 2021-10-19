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
 * \file np_broadcast_reduce_op_value_broadcast_to.cc
 * \brief CPU Implementation of broadcast and reduce functions based on value.
 */

#if MXNET_USE_TVM_OP
#include "../tvmop/op_module.h"
#endif  // MXNET_USE_TVM_OP

#include "np_broadcast_reduce_op_value.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_broadcast_to)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"array"};
                                     })
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
    .set_attr<nnvm::FGradient>("FGradient",
                               [](const nnvm::ObjectPtr& n,
                                  const std::vector<nnvm::NodeEntry>& ograds) {
                                 return MakeNonlossGradNode(
                                     "_backward_np_broadcast_to", n, ograds, {}, n->attrs.dict);
                               })
    .add_argument("array", "NDArray-or-Symbol", "The input")
    .set_attr_parser(ParamParser<BroadcastToParam>)
    .add_arguments(BroadcastToParam::__FIELDS__())
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyBroadcastToShape)
    .set_attr<FCompute>("FCompute<cpu>", NumpyBroadcastToForward<cpu>);

NNVM_REGISTER_OP(_backward_np_broadcast_to)
    .set_attr_parser(ParamParser<BroadcastToParam>)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>", NumpyBroadcastToBackward<cpu>)
    .set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs) {
      return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
    });

}  // namespace op
}  // namespace mxnet
