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
 *  Copyright (c) 2020 by Contributors
 * \file np_cross.cc
 * \brief CPU Implementation of numpy-compatible cross
 */

#include "./np_cross-inl.h"

namespace mxnet {
namespace op {

inline bool NumpyCrossShape(const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector *in_attrs,
                            mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& a_shape = in_attrs->at(0);
  const mxnet::TShape& b_shape = in_attrs->at(1);
  if (!ndim_is_known(a_shape) || !ndim_is_known(b_shape)) {
    return false;
  }

  if (shape_is_known(a_shape) && shape_is_known(b_shape)) {
    const NumpyCrossParam& param = nnvm::get<NumpyCrossParam>(attrs.parsed);
    const int a_ndim = a_shape.ndim();
    const int b_ndim = b_shape.ndim();
    CHECK_GE(a_ndim, 1) << "Array must be at least one-dimensional";
    CHECK_GE(b_ndim, 1) << "Array must be at least one-dimensional";
    CHECK_LE(a_ndim, broadcast::MAX_DIM)
      << "cross product support at most " << broadcast::MAX_DIM << " dimensions";
    CHECK_LE(b_ndim, broadcast::MAX_DIM)
      << "cross product support at most " << broadcast::MAX_DIM << " dimensions";

    const Tuple<int> a_moveaxis_index = GetMoveaxisIndex(param.axisa, -1, a_shape);
    const Tuple<int> b_moveaxis_index = GetMoveaxisIndex(param.axisb, -1, b_shape);
    const mxnet::TShape a_moveaxis_shape = GetMoveaxisShape(a_moveaxis_index, a_shape);
    const mxnet::TShape b_moveaxis_shape = GetMoveaxisShape(b_moveaxis_index, b_shape);

    CHECK(a_moveaxis_shape[a_ndim - 1] == 2 || a_moveaxis_shape[a_ndim - 1] == 3)
      << "incompatible dimensions for cross product and axis should have dimensions 2 or 3.";
    CHECK(b_moveaxis_shape[b_ndim - 1] == 2 || b_moveaxis_shape[b_ndim - 1] == 3)
      << "incompatible dimensions for cross product and axis should have dimensions 2 or 3.";

    if (a_ndim == 1 && b_ndim == 1) {
      if (a_moveaxis_shape[a_ndim - 1] == 2 && b_moveaxis_shape[b_ndim - 1] == 2) {
        // Both 1-D arrays with dim = 2, cross product of vectors.
        SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(0, 0));
      } else {
        // Both 1-D arrays with at least one dim = 3, cross product of vectors.
        SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(1, 3));
      }
    } else {
      mxnet::TShape c_shape;
      GetOrCheckLRShape(attrs, a_moveaxis_shape, b_moveaxis_shape, &c_shape);
      if (a_moveaxis_shape[a_ndim - 1] == 2 && b_moveaxis_shape[b_ndim - 1] == 2) {
        // At least one N-D arrays and both dim = 2, param.axisc is ignored.
        SHAPE_ASSIGN_CHECK(*out_attrs, 0, c_shape);
      } else {
        // At least one N-D arrays and at least one dim = 3, param.axisc not ignored.
        // Check axisc is within bounds.
        const Tuple<int> c_moveaxis_index = GetMoveaxisIndex(-1, param.axisc, c_shape);
        const mxnet::TShape c_moveaxis_shape = GetMoveaxisShape(c_moveaxis_index, c_shape);
        SHAPE_ASSIGN_CHECK(*out_attrs, 0, c_moveaxis_shape);
      }
    }
  }
  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

DMLC_REGISTER_PARAMETER(NumpyCrossParam);

NNVM_REGISTER_OP(_npi_cross)
.set_attr_parser(ParamParser<NumpyCrossParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"a", "b"};
})
.set_attr<mxnet::FInferShape>("FInferShape", NumpyCrossShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs){
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", NumpyCrossForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_cross"})
.add_argument("a", "NDArray-or-Symbol", "First vector")
.add_argument("b", "NDArray-or-Symbol", "Second vector")
.add_arguments(NumpyCrossParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_npi_cross)
.set_attr_parser(ParamParser<NumpyCrossParam>)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest>(1, ResourceRequest::kTempSpace);
})
.set_attr<FCompute>("FCompute<cpu>", NumpyCrossBackward<cpu>);

}  // namespace op
}  // namespace mxnet
