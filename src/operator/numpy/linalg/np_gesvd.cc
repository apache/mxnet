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
 * \file np_svd.cc
 * \brief CPU implementation of the SVD Operator
 */
#include <mxnet/operator_util.h>
#include <vector>
#include "./np_gesvd-inl.h"
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../elemwise_op_common.h"

namespace mxnet {
namespace op {

// Shape inference function for gesvd
// Inputs: A. Outputs: UT, L, V
inline bool NumpyLaGesvdShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector* in_attrs,
                           mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 3);
  const mxnet::TShape& in_a = (*in_attrs)[0];
  const mxnet::TShape& out_ut = (*out_attrs)[0];
  const mxnet::TShape& out_l = (*out_attrs)[1];
  const mxnet::TShape& out_v = (*out_attrs)[2];
  if (in_a.ndim() >= 2) {
    // Forward shape inference.
    const int ndim(in_a.ndim());
    CHECK_LE(in_a[ndim - 2], in_a[ndim - 1])
      << "Input A shape wrong: The second to last dimension must be less than or"
      << "equal to the last dimension";
    // V must have same shape as A
    SHAPE_ASSIGN_CHECK(*out_attrs, 2, in_a);
    std::vector<int> oshape_l(ndim - 1);
    for (int i = 0; i < ndim - 1; ++i) {
      oshape_l[i] = in_a[i];
    }
    mxnet::TShape tshape_l(oshape_l.begin(), oshape_l.end());
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, tshape_l);
    std::vector<int> oshape_ut(ndim);
    for (int i = 0; i < ndim - 1; ++i) {
      oshape_ut[i] = in_a[i];
    }
    oshape_ut[ndim - 1] = in_a[ndim - 2];
    mxnet::TShape tshape_ut(oshape_ut.begin(), oshape_ut.end());
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, tshape_ut);
    return true;
  }
  if (out_ut.ndim() >= 2 && out_ut.ndim() == out_l.ndim()+1 &&
      out_v.ndim() == out_ut.ndim()) {
    // Backward shape inference.
    const int ndim(out_ut.ndim());
    for (int i = 0; i < ndim - 1; ++i) {
      CHECK_EQ(out_ut[i], out_l[i])
        << "Outputs UT, L must have same dimensions except for last";
      CHECK_EQ(out_v[i], out_l[i])
        << "Outputs V, L must have same dimensions except for last";
    }
    CHECK_EQ(out_ut[ndim - 2], out_ut[ndim - 1])
      << "Output UT shape wrong: Last two dimensions must be equal";
    CHECK_LE(out_v[ndim-2], out_v[ndim-1])
      << "Output V shape wrong: The second to last dimension must be less than or"
      << "equal to the last dimension";
    // A must have same shape as V
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_v);
    return true;
  }
  return false;
}

NNVM_REGISTER_OP(_npi_svd)
.describe(R"code()code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(3)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"A"}; })
.set_attr<mxnet::FInferShape>("FInferShape", NumpyLaGesvdShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 3>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs) {
  return std::vector<std::pair<int, int>>{{0, 2}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<FCompute>("FCompute<cpu>", NumpyLaGesvdForward<cpu, gesvd>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_npi_svd"})
.add_argument("A", "NDArray-or-Symbol", "Input matrices to be factorized");

NNVM_REGISTER_OP(_backward_npi_svd)
.set_num_inputs(6)
.set_num_outputs(1)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs) {
  return std::vector<std::pair<int, int> >{{2, 0}}; })
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", NumpyLaGesvdBackward<cpu, gesvd_backward>);

}  // namespace op
}  // namespace mxnet
