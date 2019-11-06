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
 * \file np_polyval_op.cc
 * \brief Implement polyval on cpu
 */

#ifdef MXNET_USE_TVM_OP

#include "np_polyval_op-inl.h"
#include <vector>

namespace mxnet {
namespace op {

inline bool TVMPolyvalShape(const nnvm::NodeAttrs& attrs,
                     mxnet::ShapeVector *in_attrs,
                     mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& p_shape = in_attrs->at(0);
  const mxnet::TShape& x_shape = in_attrs->at(1);
  const mxnet::TShape& v_shape = out_attrs->at(0);
  CHECK_EQ(p_shape.ndim(), 1U);
  CHECK_EQ(x_shape.ndim(), 1U);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, x_shape);
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, v_shape);
  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

NNVM_REGISTER_OP(_npi_polyval)
.set_num_inputs(2)
.set_num_outputs(1)
.add_argument("p", "NDArray-or-Symbol", "polynomial coefficients")
.add_argument("x", "NDArray-or-Symbol", "variables")
.set_attr<nnvm::FListInputNames>("FListInputNames",
[](const NodeAttrs& attrs) {
  return std::vector<std::string>{"p", "x"};
})
.set_attr<mxnet::FInferShape>("FInferShape", TVMPolyvalShape)
.set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
.set_attr<mxnet::FCompute>("FCompute<cpu>", TVMPolyvalForward<false>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_polyval"});

NNVM_REGISTER_OP(_backward_npi_polyval)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<mxnet::FCompute>("FCompute<cpu>", TVMPolyvalBackward<false>);

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_TVM_OP

