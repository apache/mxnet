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
 * Copyright (c) 2020 by Contributors
 * \file np_qr.cc
 * \brief CPU implementation of the QR Operator
 */
#include <mxnet/operator_util.h>
#include <vector>
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../elemwise_op_common.h"
#include "./np_qr-inl.h"

namespace mxnet {
namespace op {

// Shape inference function for qr
// Inputs: A. Outputs: Q, R
inline bool NumpyLaQrShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector* in_attrs,
                           mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 2U);
  const mxnet::TShape& in_a = (*in_attrs)[0];

  if (in_a.ndim() >= 2) {
    // Forward shape inference.
    const int ndim(in_a.ndim());
    const int k = in_a[ndim - 2] > in_a[ndim - 1] ? in_a[ndim - 1] : in_a[ndim - 2];
    // Q
    std::vector<int> oshape_q(ndim);
    for (int i = 0; i < ndim - 1; ++i) {
      oshape_q[i] = in_a[i];
    }
    oshape_q[ndim - 1] = k;
    mxnet::TShape tshape_q(oshape_q.begin(), oshape_q.end());
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, tshape_q);
    // R
    std::vector<int> oshape_r(ndim);
    for (int i = 0; i < ndim - 2; ++i) {
      oshape_r[i] = in_a[i];
    }
    oshape_r[ndim - 2] = k;
    oshape_r[ndim - 1] = in_a[ndim - 1];
    mxnet::TShape tshape_r(oshape_r.begin(), oshape_r.end());
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, tshape_r);
    return true;
  }
  return false;
}

inline bool NumpyLaQrType(const nnvm::NodeAttrs& attrs,
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

NNVM_REGISTER_OP(_npi_qr)
.describe(R"code()code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"A"}; })
.set_attr<mxnet::FInferShape>("FInferShape", NumpyLaQrShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyLaQrType)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<FCompute>("FCompute<cpu>", NumpyLaQrForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_npi_qr"})
.add_argument("A", "NDArray-or-Symbol", "Input matrices to be factorized");

NNVM_REGISTER_OP(_backward_npi_qr)
.set_num_inputs(5)
.set_num_outputs(1)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", NumpyLaQrBackward<cpu>);

}  // namespace op
}  // namespace mxnet
