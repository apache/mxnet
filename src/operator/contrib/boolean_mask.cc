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

bool BooleanMaskStorageType(const nnvm::NodeAttrs& attrs,
                            const int dev_mask,
                            DispatchMode* dispatch_mode,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  for (size_t i = 0; i < out_attrs->size(); i++)
    out_attrs->at(i) = kDefaultStorage;
  *dispatch_mode = DispatchMode::kFComputeEx;
  return true;
}

NNVM_REGISTER_OP(_contrib_BooleanMask)
.describe(R"code(
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<BooleanMaskParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<FComputeEx>("FComputeEx<cpu>", BooleanMaskForward<cpu>)
.set_attr<FInferStorageType>("FInferStorageType", BooleanMaskStorageType)
//.set_attr<nnvm::FGradient>("FGradient",
//  ElemwiseGradUseNone{"_backward_contrib_BooleanMask"})
.add_argument("data", "NDArray-or-Symbol", "Data")
.add_argument("index", "NDArray-or-Symbol", "Mask")
.add_arguments(BooleanMaskParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
