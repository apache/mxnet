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
 * \file boolean_mask.cc
*/

#include "./boolean_mask-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(BooleanMaskParam);


bool BooleanMaskType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_attrs,
                     std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}

bool BooleanMaskStorageType(const nnvm::NodeAttrs& attrs,
                            const int dev_mask,
                            DispatchMode* dispatch_mode,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
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

bool BooleanMaskBackStorageType(const nnvm::NodeAttrs& attrs,
                                const int dev_mask,
                                DispatchMode* dispatch_mode,
                                std::vector<int> *in_attrs,
                                std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3);
  CHECK_EQ(out_attrs->size(), 2);
  for (int &attr : *in_attrs) {
    CHECK_EQ(attr, kDefaultStorage) << "Only default storage is supported";
  }
  for (int &attr : *out_attrs) {
    attr = kDefaultStorage;
  }
  for (size_t i = 0; i < out_attrs->size(); i++)
    out_attrs->at(i) = kDefaultStorage;
  *dispatch_mode = DispatchMode::kFComputeEx;
  return true;
}

NNVM_REGISTER_OP(_contrib_boolean_mask)
.describe(R"code(
Experimental CPU-only support for boolean masking.
Given an n-d NDArray data, and a 1-d NDArray index,
the operator produces an un-predeterminable shaped n-d NDArray out,
which stands for the rows in x where the corresonding element in index is non-zero.

>>> data = mx.nd.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
>>> index = mx.nd.array([0, 1, 0])
>>> out = mx.nd.contrib.boolean_mask(data, index)
>>> out

[[4. 5. 6.]]
<NDArray 1x3 @cpu(0)>

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<BooleanMaskParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FInferType>("FInferType", BooleanMaskType)
.set_attr<FComputeEx>("FComputeEx<cpu>", BooleanMaskForward<cpu>)
.set_attr<FInferStorageType>("FInferStorageType", BooleanMaskStorageType)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_contrib_boolean_mask"})
.add_argument("data", "NDArray-or-Symbol", "Data")
.add_argument("index", "NDArray-or-Symbol", "Mask")
.add_arguments(BooleanMaskParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_contrib_boolean_mask)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", BooleanMaskBackStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", BooleanMaskBackward<cpu>)
.add_arguments(BooleanMaskParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
