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
 * \file np_dot.cc
 * \brief CPU Implementation of numpy-compatible dot
 */

#include "./np_dot-inl.h"

namespace mxnet {
namespace op {

inline bool NumpyDotShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector *in_attrs,
                          mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& a_shape = in_attrs->at(0);
  const mxnet::TShape& b_shape = in_attrs->at(1);

  if (!shape_is_known(a_shape) || !shape_is_known(b_shape)) {
    return false;
  }

  if (a_shape.ndim() == 1 && b_shape.ndim() == 1) {
    // Case 1: both 1-D arrays, inner product of vectors
    CHECK_EQ(a_shape[0], b_shape[0]);
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(0, 0));
  } else if (a_shape.ndim() == 2 && b_shape.ndim() == 2) {
    // Case 2: both 2-D arrays, matrix multiplication
    CHECK_EQ(a_shape[1], b_shape[0]);
    mxnet::TShape mm_shape(2, 0);
    mm_shape[0] = a_shape[0];
    mm_shape[1] = b_shape[1];
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mm_shape);
  } else if (a_shape.ndim() == 0 || b_shape.ndim() == 0) {
    // Case 3 + 3.5: either of them is a scalar, just scale by one of them
    mxnet::TShape oshape = (a_shape.ndim() == 0) ? b_shape : a_shape;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  } else if (b_shape.ndim() == 1) {
    // Case 4: a is N-D array and b is 1-D array, sum product over the last axis
    CHECK_EQ(a_shape[a_shape.ndim() - 1], b_shape[0]);
    mxnet::TShape out_shape(a_shape.ndim() - 1, 0);
    for (int i = 0; i < a_shape.ndim() - 1; ++i) {
      out_shape[i] = a_shape[i];
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
  } else {
    // Case 5: a is N-D array and b is M-D array, sum product over the last axis
    //         of a and the 2nd-to-last axis of b
    LOG(FATAL) << "Case 5 not implemented yet...";
  }
  return true;
}

NNVM_REGISTER_OP(_numpy_dot)
.describe(R"doc(Dot product of two arrays. Specifically,

- If both a and b are 1-D arrays, it is inner product of vectors.

- If both a and b are 2-D arrays, it is matrix multiplication.

- If either a or b is 0-D (scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.

- If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.

- If a is an N-D array and b is an M-D array (where M>=2), it is a sum product over the last axis of a and the second-to-last axis of b:

  Example ::

    dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

)doc" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "b"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", NumpyDotShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyDotForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_np_dot"})
.add_argument("a", "NDArray-or-Symbol", "First input")
.add_argument("b", "NDArray-or-Symbol", "Second input");

NNVM_REGISTER_OP(_backward_np_dot)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyDotBackward<cpu>);

}  // namespace op
}  // namespace mxnet
