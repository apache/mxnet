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
 * Copyright (c) 2019 by Contributors
 * \file np_lstsq.cc
 * \brief CPU implementation of the lstsq Operator
 */
#include "./np_lstsq-inl.h"

namespace mxnet {
namespace op {

inline bool LstsqOpStorageType(const nnvm::NodeAttrs& attrs,
                               const int dev_mask,
                               DispatchMode* dispatch_mode,
                               std::vector<int> *in_attrs,
                               std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  for (int& attr : *in_attrs) {
    CHECK_EQ(attr, kDefaultStorage) << "Only default storage is supported";
  }
  for (int& attr : *out_attrs) {
    attr = kDefaultStorage;
  }
  *dispatch_mode = DispatchMode::kFComputeEx;
  return true;
}

inline bool LstsqOpType(const nnvm::NodeAttrs& attrs,
                        std::vector<int>* in_attrs,
                        std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 4U);
  const int a_type = in_attrs->at(0);
  const int b_type = in_attrs->at(1);
  CHECK(a_type == mshadow::kFloat32 || a_type == mshadow::kFloat64)
    << "lstsq operation only supports 32-bit and 64-bit floating point";
  CHECK(b_type == mshadow::kFloat32 || b_type == mshadow::kFloat64)
    << "lstsq operation only supports 32-bit and 64-bit floating point";

  if (mshadow::kFloat32 == a_type && mshadow::kFloat32 == b_type) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
    TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat32);
    TYPE_ASSIGN_CHECK(*out_attrs, 2, mshadow::kInt32);
    TYPE_ASSIGN_CHECK(*out_attrs, 3, mshadow::kFloat32);
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat64);
    TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat64);
    TYPE_ASSIGN_CHECK(*out_attrs, 2, mshadow::kInt32);
    TYPE_ASSIGN_CHECK(*out_attrs, 3, mshadow::kFloat64);
  }
  return out_attrs->at(0) != -1 &&
         out_attrs->at(1) != -1 &&
         out_attrs->at(2) != -1 &&
         out_attrs->at(3) != -1;
}

DMLC_REGISTER_PARAMETER(LstsqParam);

NNVM_REGISTER_OP(_npi_lstsq)
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(mxnet::op::ParamParser<LstsqParam>)
.set_num_inputs(2)
.set_num_outputs(4)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs){
  return std::vector<std::string>{"A", "B"};
})
.set_attr<nnvm::FInferType>("FInferType", LstsqOpType)
.set_attr<FInferStorageType>("FInferStorageType", LstsqOpStorageType)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs){
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FComputeEx>("FComputeEx<cpu>", LstsqOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("A", "NDArray-or-Symbol", "Tensor of matrix")
.add_argument("B", "NDArray-or-Symbol", "Tensor of matrix")
.add_arguments(LstsqParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
