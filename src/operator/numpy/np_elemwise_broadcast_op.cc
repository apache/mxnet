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

#include "./np_elemwise_broadcast_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyBinaryScalarParam);

#define MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(name)                    \
  NNVM_REGISTER_OP(name)                                                  \
  .set_num_inputs(1)                                                      \
  .set_num_outputs(1)                                                     \
  .set_attr_parser(ParamParser<NumpyBinaryScalarParam>)                   \
  .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)       \
  .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryScalarType)        \
  .set_attr<FResourceRequest>("FResourceRequest",                         \
    [](const NodeAttrs& attrs) {                                          \
      return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};   \
    })                                                                    \
  .add_argument("data", "NDArray-or-Symbol", "source input")              \
  .add_arguments(NumpyBinaryScalarParam::__FIELDS__())

bool NumpyBinaryMixedPrecisionType(const nnvm::NodeAttrs& attrs,
                                   std::vector<int>* in_attrs,
                                   std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int ltype = in_attrs->at(0);
  const int rtype = in_attrs->at(1);
  if (ltype != -1 && rtype != -1 && (ltype != rtype)) {
    // Only when both input types are known and not the same, we enter the mixed-precision mode
    TYPE_ASSIGN_CHECK(*out_attrs, 0, common::np_binary_out_infer_type(ltype, rtype));
  } else {
    return ElemwiseType<2, 1>(attrs, in_attrs, out_attrs);
  }
  return true;
}

#define MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(name)                \
  NNVM_REGISTER_OP(name)                                                       \
  .set_num_inputs(2)                                                           \
  .set_num_outputs(1)                                                          \
  .set_attr<nnvm::FListInputNames>("FListInputNames",                          \
    [](const NodeAttrs& attrs) {                                               \
      return std::vector<std::string>{"lhs", "rhs"};                           \
    })                                                                         \
  .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)           \
  .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryMixedPrecisionType)     \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                            \
    [](const NodeAttrs& attrs){                                                \
      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};                \
    })                                                                         \
  .set_attr<FResourceRequest>("FResourceRequest",                              \
    [](const NodeAttrs& attrs) {                                               \
      return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};        \
    })                                                                         \
  .add_argument("lhs", "NDArray-or-Symbol", "First input to the function")     \
  .add_argument("rhs", "NDArray-or-Symbol", "Second input to the function")

MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(_npi_add)
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastComputeWithBool<cpu, op::mshadow_op::plus, op::mshadow_op::mixed_plus,
                                      op::mshadow_op::mixed_plus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_broadcast_add"});

NNVM_REGISTER_OP(_backward_npi_broadcast_add)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyBinaryBackwardUseIn<cpu, mshadow_op::posone,
                                                                mshadow_op::posone>);

MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(_npi_subtract)
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastCompute<cpu, op::mshadow_op::minus, op::mshadow_op::mixed_minus,
                              op::mshadow_op::mixed_rminus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_broadcast_sub"});

NNVM_REGISTER_OP(_backward_npi_broadcast_sub)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyBinaryBackwardUseIn<cpu, mshadow_op::posone,
                                                                mshadow_op::negone>);

MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(_npi_multiply)
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastComputeWithBool<cpu, op::mshadow_op::mul, op::mshadow_op::mixed_mul,
                                      op::mshadow_op::mixed_mul>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_broadcast_mul"});

NNVM_REGISTER_OP(_backward_npi_broadcast_mul)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyBinaryBackwardUseIn<cpu, mshadow_op::right,
                                                              mshadow_op::left>);

MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(_npi_mod)
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastCompute<cpu, op::mshadow_op::mod, op::mshadow_op::mixed_mod,
                                      op::mshadow_op::mixed_rmod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_broadcast_mod"});

NNVM_REGISTER_OP(_backward_npi_broadcast_mod)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyBinaryBackwardUseIn<cpu, mshadow_op::mod_grad,
                                                              mshadow_op::mod_rgrad>);

MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(_npi_power)
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastComputeWithBool<cpu, op::mshadow_op::power, op::mshadow_op::mixed_power,
                                      op::mshadow_op::mixed_rpower>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_broadcast_power"});

NNVM_REGISTER_OP(_backward_npi_broadcast_power)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyBinaryBackwardUseIn<cpu, mshadow_op::power_grad,
                                                              mshadow_op::power_rgrad>);

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

}  // namespace op
}  // namespace mxnet
