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
 * \file broadcast_reduce_prod_value.cc
 * \brief CPU Implementation of broadcast and reduce of prod functions based on value.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {

MXNET_OPERATOR_REGISTER_REDUCE(prod)
.describe(get_reduce_axes_description("product", __LINE__))
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow_op::product>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{ "_backward_prod" });

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_prod)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut< cpu, mshadow_op::rdiv>);

MXNET_OPERATOR_REGISTER_REDUCE(nanprod)
.describe(R"code(Computes the product of array elements over given axes treating Not a Numbers (``NaN``) as one.

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow_op::nanprod>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{ "_backward_nanprod" });

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_nanprod)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut<cpu, mshadow_op::nanprod_grad>);

}  // namespace op
}  // namespace mxnet
