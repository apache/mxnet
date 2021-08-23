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
 * \file np_tensorinv.cc
 * \brief CPU implementation placeholder of Tensor Inverse Operator
 */
#include "./np_tensorinv-inl.h"

namespace mxnet {
namespace op {

inline bool TensorinvOpShape(const nnvm::NodeAttrs &attrs,
                             std::vector<mxnet::TShape> *in_attrs,
                             std::vector<mxnet::TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& a_shape = (*in_attrs)[0];
  const int a_ndim = a_shape.ndim();
  mxnet::TShape inv_a_shape(a_shape);
  if (!ndim_is_known(a_shape)) {
    return false;
  }
  // ind > 0, defalut = 2
  int ind = 2;
  ind = nnvm::get<TensorinvParam>(attrs.parsed).ind;
  CHECK_GT(ind, 0) << "Invalid ind argument.";

  if (a_ndim > 0 && ind < a_ndim) {
    for (int i = 0; i < ind; ++i) {
      inv_a_shape[a_ndim - ind + i] = a_shape[i];
    }
    for (int i = ind; i < a_ndim; ++i) {
      inv_a_shape[i - ind] = a_shape[i];
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, inv_a_shape);
  } else {  // ind >= a_ndim
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, inv_a_shape);
  }
  CHECK_NE(inv_a_shape.ndim(), 0)
    << "can not reshape array";

  dim_t prod_front = 1, prod_back = 1;
  if (ind < a_ndim) {
    for (int i = 0; i < ind; ++i) {
      prod_front *= a_shape[i];
    }
    for (int i = ind; i < a_ndim; ++i) {
      prod_back *= a_shape[i];
    }
    CHECK_GT(prod_back, 0)
      << "can not reshape array of size 0 into shape";
  } else {
    for (int i = 0; i < a_ndim; ++i) {
      prod_front *= a_shape[i];
    }
  }
  // prod_back >= 1 and prod_front == prod_back
  CHECK_EQ(prod_front, prod_back)
    << "a shape must be square, i. e., prod(a.shape[:ind]) == prod(a.shape[ind:]).";
  return !mxnet::op::shape_is_none(out_attrs->at(0));
}

inline bool TensorinvOpType(const nnvm::NodeAttrs& attrs,
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

DMLC_REGISTER_PARAMETER(TensorinvParam);

NNVM_REGISTER_OP(_npi_tensorinv)
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(mxnet::op::ParamParser<TensorinvParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
})
.set_attr<mxnet::FInferShape>("FInferShape", TensorinvOpShape)
.set_attr<nnvm::FInferType>("FInferType", TensorinvOpType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>(1, ResourceRequest::kTempSpace);
})
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", TensorinvOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", mxnet::op::ElemwiseGradUseOut{"_backward_npi_tensorinv"})
.add_argument("a", "NDArray-or-Symbol", "First input")
.add_arguments(TensorinvParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_npi_tensorinv)
.set_attr_parser(mxnet::op::ParamParser<TensorinvParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& ){
    return std::vector<ResourceRequest>{1, ResourceRequest::kTempSpace};
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", TensorinvOpBackward<cpu>);

}  // namespace op
}  // namespace mxnet
