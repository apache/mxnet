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
 *  Copyright (c) 2019 by Contributors
 * \file np_broadcast_reduce_op_boolean.cc
 * \brief CPU Implementation of broadcast and reduce functions based on boolean.
 */

#include "./np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {

inline bool NumpyReduceAxesBoolType(const nnvm::NodeAttrs& attrs,
                                    std::vector<int> *in_attrs,
                                    std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kBool);
  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

DMLC_REGISTER_PARAMETER(NumpyReduceAxesBoolParam);

NNVM_REGISTER_OP(_npi_any)
.add_alias("_np_sometrue")
.set_attr_parser(ParamParser<NumpyReduceAxesBoolParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyReduceAxesBoolShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyReduceAxesBoolType)
.set_attr<FCompute>("FCompute<cpu>", NumpyReduceAxesBoolCompute<cpu,
  mshadow_op::sum, mshadow_op::NonZero, 0>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(NumpyReduceAxesBoolParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_all)
.set_attr_parser(ParamParser<NumpyReduceAxesBoolParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyReduceAxesBoolShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyReduceAxesBoolType)
.set_attr<FCompute>("FCompute<cpu>", NumpyReduceAxesBoolCompute<cpu,
  mshadow_op::product, mshadow_op::NonZero, 1>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(NumpyReduceAxesBoolParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
