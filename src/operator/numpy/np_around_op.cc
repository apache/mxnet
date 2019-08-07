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
 * Copyright (c) 2018 by Contributors
 * \file np_around_op.cc
*/
#include "np_around_op.h"

namespace mxnet {
namespace op {

inline bool AroundOpType(const nnvm::NodeAttrs& attrs,
                         std::vector<int>* in_attrs,
                         std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));

  if(in_attrs->at(0) == mshadow::kFloat16){
    std::ostringstream os;
    os << "Do not support `float16` as input.\n";
    throw ::mxnet::op::InferTypeError(os.str(), 0);
  }
  return out_attrs->at(0) != -1;
}

DMLC_REGISTER_PARAMETER(AroundParam);

NNVM_REGISTER_OP(_npi_around)
.set_attr_parser(ParamParser<AroundParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"x"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", AroundOpType)
.set_attr<FCompute>("FCompute<cpu>", AroundOpForward<cpu>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("x", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(AroundParam::__FIELDS__())
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

} // namespace op
} // namespace mxnet