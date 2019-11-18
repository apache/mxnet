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
 * \file np_init_op.cc
 * \brief CPU Implementation of numpy init op
 */

#include "../tensor/init_op.h"
#include "../tensor/elemwise_unary_op.h"
#include "./np_init_op.h"

namespace mxnet {
namespace op {


DMLC_REGISTER_PARAMETER(NumpyEyeParam);
DMLC_REGISTER_PARAMETER(IndicesOpParam);
DMLC_REGISTER_PARAMETER(LogspaceParam);

inline bool NumpyIndicesShape(const nnvm::NodeAttrs& attrs,
                              mxnet::ShapeVector* in_shapes,
                              mxnet::ShapeVector* out_shapes) {
  const IndicesOpParam& param = nnvm::get<IndicesOpParam>(attrs.parsed);
  CHECK_EQ(in_shapes->size(), 0U);
  CHECK_EQ(out_shapes->size(), 1U);
  CHECK_GE(param.dimensions.ndim(), 0)
    << "_npi_indices dimensions the number of dim must not be less than 0";
  mxnet::TShape param_dim = param.dimensions;
  if (!shape_is_known(param_dim)) return false;
  const int indim = param.dimensions.ndim();
  mxnet::TShape ret(indim + 1, -1);
  ret[0] = indim;
  for (int i = 1; i < indim + 1; ++i) {
    ret[i] = param.dimensions[i-1];
  }
  SHAPE_ASSIGN_CHECK(*out_shapes, 0, ret);
  return shape_is_known(out_shapes->at(0));
}

inline bool LogspaceShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector *in_attrs,
                          mxnet::ShapeVector *out_attrs) {
  const LogspaceParam& param = nnvm::get<LogspaceParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_GE(param.num, 0)
    << "Number of sequence should be non-negative, received " << param.num;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape({static_cast<nnvm::dim_t>(param.num)}));
  return true;
}

NNVM_REGISTER_OP(_npi_zeros)
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<mxnet::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType<InitOpParam>)
.set_attr<FInferStorageType>("FInferStorageType", InitStorageType<InitOpParam, true, true>)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 0>)
.add_arguments(InitOpParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_ones)
.describe("Return a new array of given shape, type, and context, filled with ones.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<mxnet::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType<InitOpParam>)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 1>)
.add_arguments(InitOpParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_identity)
.describe("Return a new identity array of given shape, type, and context.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<mxnet::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType<InitOpParam>)
.set_attr<FCompute>("FCompute<cpu>", IdentityCompute<cpu>)
.add_arguments(InitOpParam::__FIELDS__());

NNVM_REGISTER_OP(_np_zeros_like)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
  [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>(1, 0);
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 0>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("a", "NDArray-or-Symbol",
              "The shape and data-type of a define these same attributes of the returned array.");

NNVM_REGISTER_OP(_np_ones_like)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
  [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>(1, 0);
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 1>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("a", "NDArray-or-Symbol",
              "The shape and data-type of a define these same attributes of the returned array.");

NNVM_REGISTER_OP(_npi_arange)
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(RangeParamParser)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyRangeShape)
.set_attr<nnvm::FInferType>("FInferType", InitType<RangeParam>)
.set_attr<FCompute>("FCompute<cpu>", RangeCompute<cpu, RangeParam>)
.add_arguments(RangeParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_eye)
.describe("Return a 2-D array with ones on the diagonal and zeros elsewhere.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyEyeParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyEyeShape)
.set_attr<nnvm::FInferType>("FInferType", InitType<NumpyEyeParam>)
.set_attr<FCompute>("FCompute<cpu>", NumpyEyeFill<cpu>)
.add_arguments(NumpyEyeParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_indices)
.describe("Return an array representing the indices of a grid.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<IndicesOpParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyIndicesShape)
.set_attr<nnvm::FInferType>("FInferType", InitType<IndicesOpParam>)
.set_attr<FCompute>("FCompute<cpu>", IndicesCompute<cpu>)
.add_arguments(IndicesOpParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_logspace)
.describe("Return numbers spaced evenly on a log scale.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LogspaceParam>)
.set_attr<mxnet::FInferShape>("FInferShape", LogspaceShape)
.set_attr<nnvm::FInferType>("FInferType", InitType<LogspaceParam>)
.set_attr<FCompute>("FCompute<cpu>", LogspaceCompute<cpu>)
.add_arguments(LogspaceParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
