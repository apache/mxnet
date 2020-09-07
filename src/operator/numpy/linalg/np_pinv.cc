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
 * \file np_pinv.cc
 * \brief CPU implementation of the PINV Operator
 */

#include "./np_pinv-inl.h"

namespace mxnet {
namespace op {

bool PinvOpShape(const nnvm::NodeAttrs& attrs,
                 mxnet::ShapeVector *in_attrs,
                 mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& a_shape = (*in_attrs)[0];
  const mxnet::TShape& rcond_shape = (*in_attrs)[1];
  const mxnet::TShape& pinv_shape = (*out_attrs)[0];
  const int a_ndim = a_shape.ndim();

  if (shape_is_known(a_shape)) {
    // Forward shape inference.
    CHECK_GE(a_ndim, 2)
      << "Array must be at least two-dimensional";
    // Calculte pinv shape.
    std::vector<int> pinv_shape_vec(a_ndim, -1);
    for (int i = 0; i < a_ndim - 2; ++i) {
      pinv_shape_vec[i] = a_shape[i];
    }
    pinv_shape_vec[a_ndim - 2] = a_shape[a_ndim - 1];
    pinv_shape_vec[a_ndim - 1] = a_shape[a_ndim - 2];
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(pinv_shape_vec.begin(), pinv_shape_vec.end()));
    // Check rcond shape.
    GetOrCheckCutoffAndLargeShape(attrs, a_shape, rcond_shape, nullptr, nullptr);
  } else {
    // Backward shape inference.
    if (shape_is_known(pinv_shape)) {
      const int pinv_ndim = pinv_shape.ndim();
      CHECK_GE(pinv_ndim, 2)
        << "Array must be at least two-dimensional";
      // Calculte 'a' shape.
      std::vector<int> a_shape_vec(pinv_ndim, -1);
      for (int i = 0; i < pinv_ndim - 2; ++i) {
        a_shape_vec[i] = pinv_shape[i];
      }
      a_shape_vec[pinv_ndim - 2] = pinv_shape[pinv_ndim - 1];
      a_shape_vec[pinv_ndim - 1] = pinv_shape[pinv_ndim - 2];
      SHAPE_ASSIGN_CHECK(*in_attrs, 0, mxnet::TShape(a_shape_vec.begin(), a_shape_vec.end()));
      // Check rcond shape.
      GetOrCheckCutoffAndLargeShape(attrs, (*in_attrs)[0], rcond_shape, nullptr, nullptr);
    }
  }
  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

inline bool PinvOpType(const nnvm::NodeAttrs& attrs,
                       std::vector<int>* in_attrs,
                       std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  int a_type = in_attrs->at(0);
  int rcond_type = in_attrs->at(1);
  // unsupport float16
  CHECK_NE(a_type, mshadow::kFloat16)
    << "array type float16 is unsupported in linalg.";
  CHECK(rcond_type == mshadow::kFloat32 || rcond_type == mshadow::kFloat64)
    << "rcond type should be float32 or float64.";
  if (mshadow::kFloat32 == a_type) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat64);
  }
  return out_attrs->at(0) != -1;
}

DMLC_REGISTER_PARAMETER(PinvParam);

NNVM_REGISTER_OP(_npi_pinv)
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(mxnet::op::ParamParser<PinvParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs){
  return std::vector<std::string>{"A", "rcond"};
})
.set_attr<mxnet::FInferShape>("FInferShape", PinvOpShape)
.set_attr<nnvm::FInferType>("FInferType", PinvOpType)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs){
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", PinvOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("A", "NDArray-or-Symbol", "Tensor of matrix")
.add_argument("rcond", "NDArray-or-Symbol", "Cutoff for small singular values.")
.add_arguments(PinvParam::__FIELDS__());

bool PinvScalarRcondOpShape(const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector *in_attrs,
                            mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& a_shape = (*in_attrs)[0];
  const mxnet::TShape& pinv_shape = (*out_attrs)[0];
  const int a_ndim = a_shape.ndim();

  if (shape_is_known(a_shape)) {
    // Forward shape inference.
    CHECK_GE(a_ndim, 2)
      << "Array must be at least two-dimensional";
    // Calculte pinv shape.
    std::vector<int> pinv_shape_vec(a_ndim, -1);
    for (int i = 0; i < a_ndim - 2; ++i) {
      pinv_shape_vec[i] = a_shape[i];
    }
    pinv_shape_vec[a_ndim - 2] = a_shape[a_ndim - 1];
    pinv_shape_vec[a_ndim - 1] = a_shape[a_ndim - 2];
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(pinv_shape_vec.begin(), pinv_shape_vec.end()));
  } else {
    // Backward shape inference.
    if (shape_is_known(pinv_shape)) {
      const int pinv_ndim = pinv_shape.ndim();
      CHECK_GE(pinv_ndim, 2)
        << "Array must be at least two-dimensional";
      // Calculte 'a' shape.
      std::vector<int> a_shape_vec(pinv_ndim, -1);
      for (int i = 0; i < pinv_ndim - 2; ++i) {
        a_shape_vec[i] = pinv_shape[i];
      }
      a_shape_vec[pinv_ndim - 2] = pinv_shape[pinv_ndim - 1];
      a_shape_vec[pinv_ndim - 1] = pinv_shape[pinv_ndim - 2];
      SHAPE_ASSIGN_CHECK(*in_attrs, 0, mxnet::TShape(a_shape_vec.begin(), a_shape_vec.end()));
    }
  }
  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

inline bool PinvScalarRcondOpType(const nnvm::NodeAttrs& attrs,
                                  std::vector<int>* in_attrs,
                                  std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  int a_type = in_attrs->at(0);
  // unsupport float16
  CHECK_NE(a_type, mshadow::kFloat16)
    << "array type float16 is unsupported in linalg.";
  if (mshadow::kFloat32 == a_type) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat64);
  }
  return out_attrs->at(0) != -1;
}

DMLC_REGISTER_PARAMETER(PinvScalarRcondParam);

NNVM_REGISTER_OP(_npi_pinv_scalar_rcond)
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(mxnet::op::ParamParser<PinvScalarRcondParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs){
  return std::vector<std::string>{"A"};
})
.set_attr<mxnet::FInferShape>("FInferShape", PinvScalarRcondOpShape)
.set_attr<nnvm::FInferType>("FInferType", PinvScalarRcondOpType)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs){
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", PinvScalarRcondOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("A", "NDArray-or-Symbol", "Tensor of matrix")
.add_arguments(PinvScalarRcondParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
