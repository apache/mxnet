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
 * \file elemwise_sum.cc
 * \brief elementwise sum operator
*/
#include "./elemwise_sum.h"

namespace mxnet {
namespace op {

struct ElementWiseSumParam : public dmlc::Parameter<ElementWiseSumParam> {
  int num_args;
  DMLC_DECLARE_PARAMETER(ElementWiseSumParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
        .describe("Number of inputs to be summed.");
  }
};

DMLC_REGISTER_PARAMETER(ElementWiseSumParam);

std::vector<nnvm::NodeEntry> ElementWiseSumGrad(
    const nnvm::NodePtr& n,
    const std::vector<nnvm::NodeEntry>& ograds) {
  // identity constraints in the beginning for easier shape inference.
  const nnvm::Op* copy_op =
      nnvm::Op::Get("identity");
  CHECK_EQ(ograds.size(), 1);
  std::vector<nnvm::NodeEntry> ret;
  nnvm::NodeEntry n_out{n, 0, 0};
  for (size_t i = 0; i < n->inputs.size(); i++) {
    nnvm::NodePtr id_node = nnvm::Node::Create();
    id_node->attrs.op = copy_op;
    id_node->inputs = {ograds[0]};
    ret.push_back(nnvm::NodeEntry{id_node, 0, 0});
  }
  return ret;
}

NNVM_REGISTER_OP(add_n)
.add_alias("ElementWiseSum")
.describe(R"doc(Adds all input arguments element-wise.

.. math::
   add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n

``add_n`` is potentially more efficient than calling ``add`` by `n` times.
)doc" ADD_FILELINE)
.set_attr_parser(ParamParser<ElementWiseSumParam>)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    uint32_t ret = dmlc::get<ElementWiseSumParam>(attrs.parsed).num_args;
    return ret;
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<ElementWiseSumParam>(attrs.parsed).num_args;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("arg") + std::to_string(i));
    }
    return ret;
  })
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<FCompute>("FCompute<cpu>", ElementWiseSumCompute<cpu>)
.set_attr<nnvm::FInplaceOption>(
    "FInplaceOption", [](const NodeAttrs& attrs) {
      return std::vector<std::pair<int, int> >{{0, 0}};
    })
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<-1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<nnvm::FGradient>("FGradient", CloneGradient{"_backward_add_n"})
.add_argument("args", "NDArray-or-Symbol[]", "Positional input arguments");



}  // namespace op
}  // namespace mxnet
