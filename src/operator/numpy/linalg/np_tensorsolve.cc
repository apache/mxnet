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
 * \file np_tensorsolve.cc
 * \brief CPU implementation placeholder of Tensor Solve Operator
 */
#include "./np_tensorsolve-inl.h"

namespace mxnet {
namespace op {

bool TensorsolveOpShape(const nnvm::NodeAttrs& attrs,
                        mxnet::ShapeVector *in_attrs,
                        mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& a_shape = in_attrs->at(0);
  const mxnet::TShape& b_shape = in_attrs->at(1);
  const int a_ndim = a_shape.ndim();
  const int b_ndim = b_shape.ndim();

  if (!ndim_is_known(a_shape) || !ndim_is_known(b_shape)) {
    return false;
  }

  if (0 == a_ndim && 0 == b_ndim) {
    // a and b is scalar
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, b_shape);
  } else if (0 == a_ndim && 0 != b_ndim) {
    // a is scalar, b is tensor
    CHECK_EQ(b_shape.Size(), 1U)
      << "a's and b's dimensions don't match";
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, a_shape);
  } else if (0 != a_ndim && 0 == b_ndim) {
    // a is tensor, a is scalar
    CHECK_EQ(a_shape.Size(), 1U)
      << "a's and b's dimensions don't match";
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, a_shape);
  } else {
    // a and b of at least 1 dimensions.
    const TensorsolveParam& param = nnvm::get<TensorsolveParam>(attrs.parsed);
    mxnet::Tuple<int> a_axes_param = param.a_axes;
    FixNegativeAxes(&a_axes_param, a_shape);

    mxnet::Tuple<int> a_axes_remained;
    mxnet::Tuple<int> a_axes;
    GetReorderedAxes(a_axes_param, &a_axes_remained, &a_axes, a_shape);
    mxnet::TShape a_transpose_shape = GetReorderedShape(a_shape, a_axes);

    // Calculate output shape
    const int temp = a_ndim > b_ndim ? b_ndim : b_ndim - a_ndim;
    int prod_front = 1, prod_back = 1;
    mxnet::TShape out_shape(a_ndim - temp > 0 ? a_ndim - temp : 0, -1);
    for (int i = 0; i < a_ndim; ++i) {
      if (i < temp) {
        prod_front *= a_transpose_shape[i];
      } else {
        prod_back *= a_transpose_shape[i];
        out_shape[i - temp] = a_transpose_shape[i];
      }
    }
    CHECK_EQ(prod_front, prod_back) << "a shape must be square.";
    CHECK_EQ(prod_back, b_shape.Size()) << "a's and b's dimensions don't match";
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
  }

  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

inline bool TensorsolveOpType(const nnvm::NodeAttrs& attrs,
                              std::vector<int>* in_attrs,
                              std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  int a_type = in_attrs->at(0);
  int b_type = in_attrs->at(1);
  // unsupport float16
  CHECK_NE(a_type, mshadow::kFloat16)
    << "array type float16 is unsupported in linalg";
  CHECK_NE(b_type, mshadow::kFloat16)
    << "array type float16 is unsupported in linalg";
  if (mshadow::kFloat32 == a_type && mshadow::kFloat32 == b_type) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(1));
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat64);
  }
  return out_attrs->at(0) != -1;
}

DMLC_REGISTER_PARAMETER(TensorsolveParam);

NNVM_REGISTER_OP(_npi_tensorsolve)
.set_attr_parser(mxnet::op::ParamParser<TensorsolveParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "b"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", TensorsolveOpShape)
.set_attr<nnvm::FInferType>("FInferType", TensorsolveOpType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>(1, ResourceRequest::kTempSpace);
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", TensorsolveOpForward<cpu, tensorsolve>)
.set_attr<nnvm::FGradient>("FGradient",
  mxnet::op::ElemwiseGradUseInOut{"_backward_npi_tensorsolve"})
.add_argument("a", "NDArray-or-Symbol", "First input")
.add_argument("b", "NDArray-or-Symbol", "Second input")
.add_arguments(TensorsolveParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_npi_tensorsolve)
.set_attr_parser(mxnet::op::ParamParser<TensorsolveParam>)
.set_num_inputs(4)
.set_num_outputs(2)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& ){
    return std::vector<ResourceRequest>{1, ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", TensorsolveOpBackward<cpu, tensorsolve_backward>);

}  // namespace op
}  // namespace mxnet
