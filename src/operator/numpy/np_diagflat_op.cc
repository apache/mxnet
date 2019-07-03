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
 * \file np_diagflat_op.cc
 * \brief CPU Implementation of numpy-compatible diagflat operator
 */

#include "./np_diagflat_op-inl.h"

namespace mxnet {
namespace op {

inline bool NumpyDiagflatOpShape(const nnvm::NodeAttrs &attrs,
                                 mxnet::ShapeVector *in_attrs,
                                 mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);  // should have only one input
  CHECK_EQ(out_attrs->size(), 1U); // should have only one output

  auto &in_attr = (*in_attrs)[0];

  // calc the diagnal length
  // should work for scalar
  dim_t diag_len = 1;
  for (auto &d:in_attr) {
    diag_len *= d;
  }

  // adjust the output diagnal length with k
  const NumpyDiagflatParam &param = nnvm::get<NumpyDiagflatParam>(attrs.parsed);
  diag_len += abs(param.k);

  mxnet::TShape tshape({diag_len, diag_len});
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, tshape);
  return true;
}

DMLC_REGISTER_PARAMETER(NumpyDiagflatParam);

NNVM_REGISTER_OP(_np_diagflat)
    .set_attr_parser(ParamParser<NumpyDiagflatParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs &attrs) {
                                       return std::vector<std::string>{"data"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyDiagflatOpShape)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
    .set_attr<FCompute>("FCompute<cpu>", NumpyDiagflatOpForward<cpu>)
    .set_attr<nnvm::FGradient>("FGradient",
                               ElemwiseGradUseNone{"_backward_np_diagflat"})
    .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
    .add_arguments(NumpyDiagflatParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_np_diagflat)
    .set_attr_parser(ParamParser<NumpyDiagflatParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>", NumpyDiagflatOpBackward<cpu>);

}  // namespace op
}  // namespace mxnet
