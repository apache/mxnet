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
 * \file np_eigvals.cc
 * \brief CPU implementation placeholder of Eigvals Operator
 */
#include "./np_eigvals-inl.h"

namespace mxnet {
namespace op {

// Inputs: A.
// Outputs: Eig.
bool EigvalsOpShape(const nnvm::NodeAttrs& attrs,
                    mxnet::ShapeVector *in_attrs,
                    mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& a_shape = (*in_attrs)[0];
  const mxnet::TShape& eig_shape = (*out_attrs)[0];

  if (shape_is_known(a_shape)) {
    // Forward shape inference.
    const int a_ndim = a_shape.ndim();
    CHECK_GE(a_ndim, 2)
      << "Array must be at least two-dimensional";
    CHECK_EQ(a_shape[a_ndim - 2], a_shape[a_ndim - 1])
      << "Input A's last two dimension must be equal";

    // Calculate eig shape.
    std::vector<int> eig_shape_vec(a_ndim - 1, -1);
    for (int i = 0; i < a_ndim - 1; ++i) {
      eig_shape_vec[i] = a_shape[i];
    }
    mxnet::TShape eig_shape(eig_shape_vec.begin(), eig_shape_vec.end());
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, eig_shape);
  } else if (shape_is_known(eig_shape)) {
    // Backward shape inference.
    const int eig_ndim = eig_shape.ndim();
    CHECK_GE(eig_ndim, 1)
      << "Outputs W must be at least one-dimensional";
    std::vector<int> a_shape_vec(eig_ndim + 1);
    for (int i = 0; i < eig_ndim; ++i) {
      a_shape_vec[i] = eig_shape[i];
    }
    a_shape_vec[eig_ndim] = eig_shape[eig_ndim - 1];
    mxnet::TShape a_shape(a_shape_vec.begin(), a_shape_vec.end());
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, a_shape);
  }
  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

inline bool EigvalsOpType(const nnvm::NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  int a_type = in_attrs->at(0);
  // unsupport float16
  CHECK_NE(a_type, mshadow::kFloat16)
    << "array type float16 is unsupported in linalg";
  if (mshadow::kFloat32 == a_type) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat64);
  }
  return out_attrs->at(0) != -1;
}

NNVM_REGISTER_OP(_npi_eigvals)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs){
  return std::vector<std::string>{"A"};
})
.set_attr<mxnet::FInferShape>("FInferShape", EigvalsOpShape)
.set_attr<nnvm::FInferType>("FInferType", EigvalsOpType)
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", EigvalsOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("A", "NDArray-or-Symbol", "Tensor of square matrix");

DMLC_REGISTER_PARAMETER(EigvalshParam);

NNVM_REGISTER_OP(_npi_eigvalsh)
.set_attr_parser(mxnet::op::ParamParser<EigvalshParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs){
  return std::vector<std::string>{"A"};
})
.set_attr<mxnet::FInferShape>("FInferShape", EigvalsOpShape)
.set_attr<nnvm::FInferType>("FInferType", EigvalsOpType)
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", EigvalshOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("A", "NDArray-or-Symbol", "Tensor of square matrix")
.add_arguments(EigvalshParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
