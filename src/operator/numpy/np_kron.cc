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
 * \file np_kron.cc
 * \brief CPU Implementation of numpy-compatible Kronecker product
 */

#include "./np_kron-inl.h"

namespace mxnet {
namespace op {

inline bool KronOpShape(const nnvm::NodeAttrs& attrs,
                        mxnet::ShapeVector *in_attrs,
                        mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& a_shape = in_attrs->at(0);
  const mxnet::TShape& b_shape = in_attrs->at(1);

  if (!ndim_is_known(a_shape) || !ndim_is_known(b_shape)) {
    return false;
  }

  mxnet::TShape out_shape(std::max(a_shape.ndim(), b_shape.ndim()), -1);
  if (a_shape.ndim() > b_shape.ndim()) {
    for (int i = 0; i < a_shape.ndim() - b_shape.ndim(); i++) {
      out_shape[i] = a_shape[i];
    }
    for (int i = a_shape.ndim() - b_shape.ndim(); i < a_shape.ndim(); i++) {
      out_shape[i] = a_shape[i] * b_shape[i - a_shape.ndim() + b_shape.ndim()];
    }
  } else {
    for (int i = 0; i < b_shape.ndim() - a_shape.ndim(); i++) {
      out_shape[i] = b_shape[i];
    }
    for (int i = b_shape.ndim() - a_shape.ndim(); i < b_shape.ndim(); i++) {
      out_shape[i] = b_shape[i] * a_shape[i - b_shape.ndim() + a_shape.ndim()];
    }
  }

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);

  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

NNVM_REGISTER_OP(_npi_kron)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "b"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", KronOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", KronOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_kron"})
.add_argument("a", "NDArray-or-Symbol", "First input")
.add_argument("b", "NDArray-or-Symbol", "Second input");

NNVM_REGISTER_OP(_backward_npi_kron)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", KronOpBackward<cpu>);

}  // namespace op
}  // namespace mxnet
