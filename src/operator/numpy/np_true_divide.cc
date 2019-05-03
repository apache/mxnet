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
 *  Copyright (c) 2019 by Contributors
 * \file np_true_divide.cc
 * \brief CPU Implementation of true_divide operator.
 */
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

template <int num_inputs>
bool TrueDivideType(const nnvm::NodeAttrs& attrs,
                    std::vector<int>* in_attrs,
                    std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(num_inputs));
  CHECK_EQ(out_attrs->size(), 1U);
  for (const int dtype : *in_attrs) {
    if (dtype == -1) return false;
  }
  if (num_inputs == 2) {
    const int lhs_dtype = in_attrs->at(0);
    const int rhs_dtype = in_attrs->at(1);
    CHECK_EQ(lhs_dtype, rhs_dtype)
        << "_true_divide currently only supports same dtype for dividend and divisor";
  }
  auto is_float = [](const int dtype) {
    return dtype == mshadow::kFloat32 || dtype == mshadow::kFloat64 || dtype == mshadow::kFloat16;
  };

  for (const int dtype : *in_attrs) {
    CHECK(is_float(dtype)) << "_true_divide currently only supports float dtype";
  }
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  return true;
}

NNVM_REGISTER_OP(_true_divide)
.describe(R"code(
Returns a true division of the inputs, element-wise.

It currently only supports dtype float16, float32, and float64.

Example::

   x = [[ 6.,  6.,  6.],
        [ 6.,  6.,  6.]]

   y = [[ 2.],
        [ 3.]]

   _true_divide(x, y) = [[ 3.,  3.,  3.],
                         [ 2.,  2.,  2.]]

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
.set_attr<nnvm::FInferType>("FInferType", TrueDivideType<2>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, op::mshadow_op::div>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_div"})
.set_attr<mxnet::TIsNumpyCompatible>("TIsNumpyCompatible", true)
.add_argument("lhs", "NDArray-or-Symbol", "Dividend array")
.add_argument("rhs", "NDArray-or-Symbol", "Divisor array");

NNVM_REGISTER_OP(_true_divide_scalar)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser([](NodeAttrs* attrs) {
    attrs->parsed = std::stod(attrs->dict["scalar"]);
  })
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", TrueDivideType<1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, op::mshadow_op::div>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_div_scalar"})
.set_attr<mxnet::TIsNumpyCompatible>("TIsNumpyCompatible", true)
.add_argument("data", "NDArray-or-Symbol", "source input")
.add_argument("scalar", "float", "scalar input");

NNVM_REGISTER_OP(_rtrue_divide_scalar)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser([](NodeAttrs* attrs) {
  attrs->parsed = std::stod(attrs->dict["scalar"]);
  })
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", TrueDivideType<1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::rdiv>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_rdiv_scalar"})
.set_attr<mxnet::TIsNumpyCompatible>("TIsNumpyCompatible", true)
.add_argument("data", "NDArray-or-Symbol", "source input")
.add_argument("scalar", "float", "scalar input");

}  // namespace op
}  // namespace mxnet
