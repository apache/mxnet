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
 * Copyright (c) 2020 by Contributors
 * \file np_interp_op.cc
 * \brief CPU Implementation of Numpy-compatible interp
*/

#include "np_interp_op-inl.h"

namespace mxnet {
namespace op {

inline bool NumpyInterpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  CHECK_GE(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const NumpyInterpParam& param = nnvm::get<NumpyInterpParam>(attrs.parsed);

  TShape oshape;
  CHECK_EQ(in_attrs->at(0).ndim(), 1U)
    << "ValueError: Data points must be 1-D array";
  CHECK_EQ(in_attrs->at(1).ndim(), 1U)
    << "ValueError: Data points must be 1-D array";
  CHECK_EQ(in_attrs->at(0)[0], in_attrs->at(1)[0])
    << "ValueError: fp and xp are not of the same length";
  oshape = param.x_is_scalar ? TShape(0, 1) : in_attrs->at(2);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return shape_is_known(out_attrs->at(0));
}

inline bool NumpyInterpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK_GE(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat64);
  return out_attrs->at(0) != -1;
}

DMLC_REGISTER_PARAMETER(NumpyInterpParam);

NNVM_REGISTER_OP(_npi_interp)
.set_num_inputs([](const NodeAttrs& attrs) {
  const NumpyInterpParam& param =
    nnvm::get<NumpyInterpParam>(attrs.parsed);
  return param.x_is_scalar ? 2 : 3;
  })
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyInterpParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyInterpShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyInterpType)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const NumpyInterpParam& param =
      nnvm::get<NumpyInterpParam>(attrs.parsed);
    return param.x_is_scalar ?
           std::vector<std::string>{"xp", "fp"} :
           std::vector<std::string>{"xp", "fp", "x"};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyInterpForward<cpu, mshadow_op::mod>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("xp", "NDArray-or-Symbol", "Input x-coordinates")
.add_argument("fp", "NDArray-or-Symbol", "Input y-coordinates")
.add_argument("x", "NDArray-or-Symbol", "Input data")
.add_arguments(NumpyInterpParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
