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
 * \file np_elemwise_binary_op.cc
 * \brief CPU Implementation of numpy-compatible element-wise binary operations
 */

#include <mxnet/base.h>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../tensor/elemwise_binary_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

inline bool BitwiseOrType(const nnvm::NodeAttrs &attrs,
                          std::vector<int> *in_attrs,
                          std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  // check two inputs have the same type
  TYPE_ASSIGN_CHECK(*in_attrs, 0, in_attrs->at(1));
  TYPE_ASSIGN_CHECK(*in_attrs, 1, in_attrs->at(0));

  if (in_attrs->at(0) >= 0 && in_attrs->at(0) <= 2) {
    // float types are not supported
    return false;
  } else {
    // infer output type from input
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  }
  return out_attrs->at(0) != -1;
}


NNVM_REGISTER_OP(_np_bitwise_or)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
    .set_attr<nnvm::FInferType>("FInferType", BitwiseOrType)
.set_attr<nnvm::FListInputNames>("FListInputNames",
                                 [](const NodeAttrs &attrs) {
                                   return std::vector<std::string>{"x1",
                                                                   "x2"};
                                 })
.set_attr<FCompute>("FCompute<cpu>",
                    BinaryBroadcastCompute<cpu, mshadow_op::bitwise_or>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("x1", "NDArray-or-Symbol", "The input array.")
.add_argument("x2", "NDArray-or-Symbol", "The input array.");

}  // namespace op
}  // namespace mxnet
