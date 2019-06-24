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
 * \file np_elemwise_binary_op.cc
 * \brief CPU Implementation of basic functions for elementwise numpy binary broadcast operator.
 */

#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

bool NumpyBinaryScalarType(const nnvm::NodeAttrs& attrs,
                           std::vector<int>* in_attrs,
                           std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int itype = in_attrs->at(0);
  if (itype == -1) return false;
  auto is_float = [](const int dtype) {
    return dtype == mshadow::kFloat32 || dtype == mshadow::kFloat64 || dtype == mshadow::kFloat16;
  };
  CHECK(is_float(itype)) << "numpy binary scalar op currently only supports float dtype";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, itype);
  return true;
}

#define MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(name)              \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(1)                                                \
  .set_num_outputs(1)                                               \
  .set_attr_parser([](NodeAttrs* attrs) {                           \
      attrs->parsed = std::stod(attrs->dict["scalar"]);             \
    })                                                              \
  .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>) \
  .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryScalarType)  \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}};             \
    })                                                              \
  .add_argument("data", "NDArray-or-Symbol", "source input")        \
  .add_argument("scalar", "float", "scalar input")


MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_add)
.describe(R"code(Add arguments element-wise with broadcasting if necessary.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   add(x, y) = [[ 1.,  1.,  1.],
                [ 2.,  2.,  2.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, op::mshadow_op::plus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_broadcast_add"});

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_subtract)
.describe(R"code(Subtract arguments element-wise with broadcasting if necessary.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   subtract(x, y) = [[ 1.,  1.,  1.],
                     [ 0.,  0.,  0.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, op::mshadow_op::minus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_broadcast_sub"});

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_multiply)
.describe(R"code(Multiply arguments with broadcasting if necessary.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   multiply(x, y) = [[ 0.,  0.,  0.],
                     [ 1.,  1.,  1.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, op::mshadow_op::mul>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_mul"});

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_mod)
.describe(R"code(Return element-wise remainder of division.
It is equivalent to the Python modulus operator``x1 % x2`` and has the same sign as the divisor x2.

Example::

   x = [[ 8.,  8.,  8.],
        [ 8.,  8.,  8.]]

   y = [[ 2.],
        [ 3.]]

   mod(x, y) = [[ 0.,  0.,  0.],
                [ 2.,  2.,  2.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::mod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_mod"});

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_power)
.describe(R"code(First array elements raised to powers from second array, element-wise.

Raise each base in x1 to the positionally-corresponding power in x2. x1 and x2 must be
broadcastable to the same shape.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   power(x, y) = [[ 2.,  2.,  2.],
                  [ 4.,  4.,  4.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::power>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_power"});

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_maximum)
.describe(R"code()code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::maximum>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_minimum)
.describe(R"code()code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::minimum>);

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_add_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, op::mshadow_op::plus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_subtract_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, op::mshadow_op::minus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_rsubtract_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::rminus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"negative"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_multiply_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, op::mshadow_op::mul>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_mul_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_mod_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::mod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mod_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_rmod_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::rmod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_rmod_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_power_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::power>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_power_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_rpower_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::rpower>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_rpower_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_maximum_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::maximum>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_maximum_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_minimum_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::minimum>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_minimum_scalar"});

}  // namespace op
}  // namespace mxnet
