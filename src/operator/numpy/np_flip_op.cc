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
 *  Copyright (c) 2015 by Contributors
 * \file np_flip_op.cc
 * \brief CPU Implementation of flip operations
 */
#include "./np_flip_op.h"
#include "../tensor/elemwise_unary_op.h"
#include "../nn/mkldnn/mkldnn_ops-inl.h"
#include "../nn/mkldnn/mkldnn_base-inl.h"
#include "../nn/mkldnn/mkldnn_slice-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(FlipParam);

NNVM_REGISTER_OP(_npi_flip)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr_parser(ParamParser<FlipParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
[](const NodeAttrs& attrs) {
  return std::vector<std::string> {"data"};
})
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest> {ResourceRequest::kTempSpace};
})
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", FlipOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_flip"})
.add_argument("data", "NDArray-or-Symbol", "Input data array")
.add_arguments(FlipParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_npi_flip)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<FlipParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest> {ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", FlipOpForward<cpu>);

}  // namespace op
}  // namespace mxnet
