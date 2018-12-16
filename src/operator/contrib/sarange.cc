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
 * \file sarange.cc
*/
#include "./sarange-inl.h"
#include "../tensor/elemwise_binary_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SArangeParam);

inline bool SArangeType(const nnvm::NodeAttrs& attrs,
                        std::vector<int> *in_attrs,
                        std::vector<int> *out_attrs) {
  const SArangeParam& param = nnvm::get<SArangeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype);
  return true;
}

bool SArangeStorageType(const nnvm::NodeAttrs& attrs,
                        const int dev_mask,
                        DispatchMode* dispatch_mode,
                        std::vector<int> *in_attrs,
                        std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  for (int &attr : *in_attrs) {
    CHECK_EQ(attr, kDefaultStorage) << "Only default storage is supported";
  }
  for (int &attr : *out_attrs) {
    attr = kDefaultStorage;
  }
  *dispatch_mode = DispatchMode::kFComputeEx;
  return true;
}

NNVM_REGISTER_OP(_contrib_sarange)
.describe(R"code(
Experimental CPU-only support for arange of symbolic input.
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<SArangeParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferType>("FInferType", SArangeType)
.set_attr<FInferStorageType>("FInferStorageType", SArangeStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SArangeForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_contrib_sarange"})
.add_argument("data", "NDArray-or-Symbol", "Data")
.add_arguments(SArangeParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_contrib_sarange)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, unary_bwd<mshadow_op::relu_grad>>)
.add_arguments(SArangeParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
