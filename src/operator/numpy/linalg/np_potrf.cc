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
 * \file np_potrf.cc
 * \brief CPU implementation placeholder of Cholesky Operator
 */
#include <mxnet/operator_util.h>
#include <vector>
#include "./np_potrf-inl.h"
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../elemwise_op_common.h"

namespace mxnet {
namespace op {

inline bool NumpyLaCholeskyShape(const nnvm::NodeAttrs& attrs,
                                 mxnet::ShapeVector* in_attrs,
                                 mxnet::ShapeVector* out_attrs) {
  const mxnet::TShape& in_shape = (*in_attrs)[0];
  CHECK_GE(in_shape.ndim(), 2)
    << "Array must be at least two-dimensional";
  return ElemwiseShape<1, 1>(attrs, in_attrs, out_attrs);
}

// calls forward and backward implemented in la_op
NNVM_REGISTER_OP(_npi_cholesky)
.describe(R"code()code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LaCholeskyParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"A"}; } )
.set_attr<mxnet::FInferShape>("FInferShape", NumpyLaCholeskyShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs)
  { return std::vector<std::pair<int, int>>{{0, 0}}; })
.set_attr<FCompute>("FCompute<cpu>", LaOpForward<cpu, 2, 2, 1, 1, potrf>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_linalg_potrf"})
.add_argument("A", "NDArray-or-Symbol", "Tensor of input matrices to be decomposed");

}  // namespace op
}  // namespace mxnet
