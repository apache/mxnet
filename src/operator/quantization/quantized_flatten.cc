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
 *  Copyright (c) 2017 by Contributors
 * \file quantized_flatten.cc
 * \brief
 */
#include <mxnet/op_attr_types.h>
#include "./quantized_flatten-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_quantized_flatten)
.set_num_inputs(3)
.set_num_outputs(3)
.set_attr<mxnet::FInferShape>("FInferShape", QuantizedFlattenShape)
.set_attr<nnvm::FInferType>("FInferType", QuantizedFlattenType)
.set_attr<FCompute>("FCompute<cpu>", QuantizedFlattenCompute<cpu>)
// TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
// will be reverted after the improvement of CachedOP is done.
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "min_data", "max_data"};
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output", "min_output", "max_output"};
  })
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {1, 1}, {2, 2}};
  })
.add_argument("data", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`")
.add_argument("min_data", "NDArray-or-Symbol", "The minimum scalar value "
  "possibly produced for the data")
.add_argument("max_data", "NDArray-or-Symbol", "The maximum scalar value "
  "possibly produced for the data");

NNVM_REGISTER_OP(Flatten)
.set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
    nnvm::ObjectPtr node = nnvm::Node::Create();
    node->attrs.op = Op::Get("_contrib_quantized_flatten");
    node->attrs.name = "quantized_" + attrs.name;
    node->attrs.dict = attrs.dict;
    if (node->op()->attr_parser != nullptr) {
      node->op()->attr_parser(&(node->attrs));
    }
    return node;
  });

}  // namespace op
}  // namespace mxnet
