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
 * \file np_eig.cc
 * \brief CPU implementation placeholder of Eig Operator
 */
#include "./np_eig-inl.h"

namespace mxnet {
namespace op {

// Inputs: A.
// Outputs: Eig, EigVec
bool EigOpShape(const nnvm::NodeAttrs& attrs,
                mxnet::ShapeVector *in_attrs,
                mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 2U);
  const mxnet::TShape& a_shape = (*in_attrs)[0];
  const mxnet::TShape& eig_shape = (*out_attrs)[0];
  const mxnet::TShape& eigv_shape = (*out_attrs)[1];

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
    // Calculate eig vec shape: must have the same shape as A
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, a_shape);
  } else {
    // Backward shape inference.
    if (shape_is_known(eig_shape) && shape_is_known(eigv_shape)) {
      const int eig_ndim = eig_shape.ndim();
      const int eigv_ndim = eigv_shape.ndim();
      CHECK_GE(eigv_ndim, 2)
        << "Outputs V must be at least two-dimensional";
      CHECK_EQ(eigv_shape[eigv_ndim - 2], eigv_shape[eigv_ndim - 1])
        << "Outputs V's last two dimension must be equal";
      CHECK_EQ(eig_ndim + 1, eigv_ndim)
        << "Outputs W, V must satisfy W.ndim == V.ndim - 1";
      for (int i = 0; i < eig_ndim; ++i) {
        CHECK_EQ(eig_shape[i], eigv_shape[i])
          << "Outputs W, V's shape dismatch";
      }
      SHAPE_ASSIGN_CHECK(*in_attrs, 0, eigv_shape);
    } else if (shape_is_known(eig_shape)) {
      const int eig_ndim = eig_shape.ndim();
      CHECK_GE(eig_ndim, 1)
        << "Outputs W must be at least one-dimensional";
      std::vector<int> eigv_shape_vec(eig_ndim + 1);
      for (int i = 0; i < eig_ndim; ++i) {
        eigv_shape_vec[i] = eig_shape[i];
      }
      eigv_shape_vec[eig_ndim] = eig_shape[eig_ndim - 1];
      mxnet::TShape eigv_shape(eigv_shape_vec.begin(), eigv_shape_vec.end());
      SHAPE_ASSIGN_CHECK(*in_attrs, 0, eigv_shape);
      SHAPE_ASSIGN_CHECK(*out_attrs, 1, eigv_shape);
    } else {
      const int eigv_ndim = eigv_shape.ndim();
      CHECK_GE(eigv_ndim, 2)
        << "Outputs V must be at least two-dimensional";
      CHECK_EQ(eigv_shape[eigv_ndim - 2], eigv_shape[eigv_ndim - 1])
        << "Outputs V's last two dimension must be equal";
      std::vector<int> eig_shape_vec(eigv_ndim - 1);
      for (int i = 0; i < eigv_ndim - 1; ++i) {
        eig_shape_vec[i] = eigv_shape[i];
      }
      mxnet::TShape eig_shape(eig_shape_vec.begin(), eig_shape_vec.end());
      SHAPE_ASSIGN_CHECK(*in_attrs, 0, eigv_shape);
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, eig_shape);
    }
  }
  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

inline bool EigOpType(const nnvm::NodeAttrs& attrs,
                      std::vector<int>* in_attrs,
                      std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 2U);
  int a_type = in_attrs->at(0);
  // unsupport float16
  CHECK_NE(a_type, mshadow::kFloat16)
    << "array type float16 is unsupported in linalg";
  if (mshadow::kFloat32 == a_type) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
    TYPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(0));
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat64);
    TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat64);
  }
  return out_attrs->at(0) != -1 && out_attrs->at(1) != -1;
}

NNVM_REGISTER_OP(_npi_eig)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs){
  return std::vector<std::string>{"A"};
})
.set_attr<mxnet::FInferShape>("FInferShape", EigOpShape)
.set_attr<nnvm::FInferType>("FInferType", EigOpType)
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", EigOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("A", "NDArray-or-Symbol", "Tensor of square matrix");

DMLC_REGISTER_PARAMETER(EighParam);

NNVM_REGISTER_OP(_npi_eigh)
.set_attr_parser(mxnet::op::ParamParser<EighParam>)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs){
  return std::vector<std::string>{"A"};
})
.set_attr<mxnet::FInferShape>("FInferShape", EigOpShape)
.set_attr<nnvm::FInferType>("FInferType", EigOpType)
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", EighOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("A", "NDArray-or-Symbol", "Tensor of real matrices")
.add_arguments(EighParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
