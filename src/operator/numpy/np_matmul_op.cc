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
 * \file np_matmul_op.cc
 * \brief CPU Implementation of numpy-compatible matmul
 */

#include <string>
#include "np_matmul_op-inl.h"

namespace mxnet {
namespace op {

inline bool NumpyMatmulShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector *in_attrs,
                             mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& a_shape = in_attrs->at(0);
  const mxnet::TShape& b_shape = in_attrs->at(1);
  const size_t a_ndim = a_shape.ndim();
  const size_t b_ndim = b_shape.ndim();
  if (!ndim_is_known(a_shape) || !ndim_is_known(b_shape)) {
    return false;
  }

  CHECK_NE(a_ndim, 0)
    << "Multiplication by scalars is not allowed.\n";
  CHECK_NE(b_ndim, 0)
    << "Multiplication by scalars is not allowed.\n";

  if (a_ndim == 1 && b_ndim == 1) {
    // case 1: both 1-D arrays, inner product of vectors
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, in_attrs->at(1));
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, in_attrs->at(0));
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(0, 0));
  } else if (a_ndim == 2 && b_ndim == 2) {
    // case 2: both 2-D arrays, matrix multiplication
    mxnet::TShape tmp_shape(2, -1);
    tmp_shape[1] = b_shape[0];
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, tmp_shape);

    tmp_shape[0] = a_shape[1];
    tmp_shape[1] = -1;
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, tmp_shape);

    tmp_shape[0] = a_shape[0];
    tmp_shape[1] = b_shape[1];
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, tmp_shape);
  } else if (b_ndim == 1) {
    // case 3: If the second argument is 1-D, it is promoted to a matrix
    //         by appending a 1 to its dimensions.
    //         After matrix multiplication the appended 1 is removed.
    TShape tmp_shape(a_ndim, -1);
    tmp_shape[a_ndim - 1] = b_shape[0];
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, tmp_shape);

    tmp_shape = TShape(1, -1);
    tmp_shape[0] = a_shape[a_ndim - 1];
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, tmp_shape);

    mxnet::TShape out_shape(a_ndim - 1, -1);
    for (size_t i = 0; i < a_ndim - 1; ++i) {
      out_shape[i] = a_shape[i];
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
  } else if (a_ndim == 1) {
    // Case 4: If the first argument is 1-D, it is promoted to a matrix
    //         by prepending a 1 to its dimensions.
    //         After matrix multiplication the prepended 1 is removed.
    TShape tmp_shape(b_ndim, -1);
    tmp_shape[b_ndim - 2] = a_shape[0];
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, tmp_shape);

    tmp_shape = TShape(1, -1);
    tmp_shape[0] = b_shape[b_ndim - 2];
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, tmp_shape);

    mxnet::TShape out_shape(b_ndim - 1, -1);
    for (size_t i = 0; i < b_ndim - 2; ++i) {
      out_shape[i] = b_shape[i];
    }
    out_shape[b_ndim - 2] = b_shape[b_ndim - 1];
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
  } else {
    // case 5: If either argument is N-D, N > 2, it is treated as a stack of matrices
    //         residing in the last two indexes and broadcast accordingly.
    TShape tmp_shape(a_ndim, -1);
    tmp_shape[a_ndim - 1] = b_shape[b_ndim - 2];
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, tmp_shape);
    tmp_shape = TShape(b_ndim, -1);
    tmp_shape[b_ndim - 2] = a_shape[a_ndim - 1];
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, tmp_shape);
    size_t ndim = std::max(a_ndim, b_ndim);
    mxnet::TShape out_shape(ndim, -1);
    out_shape[ndim - 1] = b_shape[b_ndim - 1];
    out_shape[ndim - 2] = a_shape[a_ndim - 2];
    for (int p = ndim - 3, pa = a_ndim - 3, pb = b_ndim - 3;
         p >= 0; --p, --pa, --pb) {
      if (pa >= 0 && pb >= 0) {
        if (a_shape[pa] == 1) {
          out_shape[p] = b_shape[pb];
        } else if (b_shape[pb] == 1) {
          out_shape[p] = a_shape[pa];
        } else {
          CHECK_EQ(a_shape[pa], b_shape[pb])
            << "Could not be broadcast.\n";
          out_shape[p] = b_shape[pb];
        }
      } else if (pa >= 0) {
        out_shape[p] = a_shape[pa];
      } else if (pb >= 0) {
        out_shape[p] = b_shape[pb];
      }
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
  }
  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

NNVM_REGISTER_OP(_npi_matmul)
.describe(R"doc()doc" ADD_FILELINE)
.set_num_inputs(2U)
.set_num_outputs(1U)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "b"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"out"};
})
.set_attr<mxnet::FInferShape>("FInferShape", NumpyMatmulShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", NumpyMatmulForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_np_matmul"})
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("a", "NDArray-or-Symbol", "First input")
.add_argument("b", "NDArray-or-Symbol", "Second input");

NNVM_REGISTER_OP(_backward_np_matmul)
.set_num_inputs(3U)
.set_num_outputs(2U)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyMatmulBackward<cpu>);

}  // namespace op
}  // namespace mxnet
