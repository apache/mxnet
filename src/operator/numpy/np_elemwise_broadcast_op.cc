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
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

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

#ifndef _WIN32
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
  .add_argument("lhs", "NDArray-or-Symbol", "First input to the function")     \
  .add_argument("rhs", "NDArray-or-Symbol", "Second input to the function")
#else
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
  [](const NodeAttrs& attrs) {                                                 \
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};          \
  })                                                                           \
  .add_argument("lhs", "NDArray-or-Symbol", "First input to the function")     \
  .add_argument("rhs", "NDArray-or-Symbol", "Second input to the function")
#endif


MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_mod)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::mod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_mod"});

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_power)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::power>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_power"});

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

#if MXNET_USE_TVM_OP

static constexpr char func_multiply_cpu[] = "multiply_cpu";
static constexpr char func_multiply_gpu[] = "multiply_gpu";
static constexpr char func_backward_multiply_cpu[] = "backward_multiply_cpu";
static constexpr char func_backward_multiply_gpu[] = "backward_multiply_gpu";
static constexpr char func_add_cpu[] = "add_cpu";
static constexpr char func_add_gpu[] = "add_gpu";
static constexpr char func_backward_add_cpu[] = "backward_add_cpu";
static constexpr char func_backward_add_gpu[] = "backward_add_gpu";
static constexpr char func_subtract_cpu[] = "subtract_cpu";
static constexpr char func_subtract_gpu[] = "subtract_gpu";
static constexpr char func_backward_subtract_cpu[] = "backward_subtract_cpu";
static constexpr char func_backward_subtract_gpu[] = "backward_subtract_gpu";

#define MXNET_OPERAOTR_REGISTER_BACKWARD_NP_BINARY_BROADCAST_USE_NONE(name)  \
  NNVM_REGISTER_OP(_backward_npi_##name)                                     \
  .set_num_inputs(1)                                                         \
  .set_num_outputs(2)                                                        \
  .set_attr<nnvm::TIsBackward>("TIsBackward", true)                          \
  .set_attr<FCompute>("FCompute<cpu>",                                       \
    TVMBinaryBroadcastBackwardUseNone{func_backward_##name##_cpu});

#define MXNET_OPERAOTR_REGISTER_BACKWARD_NP_BINARY_BROADCAST_USE_IN(name)  \
  NNVM_REGISTER_OP(_backward_npi_##name)                                   \
  .set_num_inputs(3)                                                       \
  .set_num_outputs(2)                                                      \
  .set_attr<nnvm::TIsBackward>("TIsBackward", true)                        \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                        \
    [](const NodeAttrs& attrs){                                            \
      return std::vector<std::pair<int, int> >{{0, 0}};                    \
    })                                                                     \
  .set_attr<FCompute>("FCompute<cpu>",                                     \
    TVMBinaryBroadcastBackwardUseIn{func_backward_##name##_cpu});

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_multiply)
.set_attr<FCompute>("FCompute<cpu>", TVMBinaryBroadcastCompute{func_multiply_cpu})
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_multiply"});

MXNET_OPERAOTR_REGISTER_BACKWARD_NP_BINARY_BROADCAST_USE_IN(multiply);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_add)
.set_attr<FCompute>("FCompute<cpu>", TVMBinaryBroadcastCompute{func_add_cpu})
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_add"});

MXNET_OPERAOTR_REGISTER_BACKWARD_NP_BINARY_BROADCAST_USE_NONE(add);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_subtract)
.set_attr<FCompute>("FCompute<cpu>", TVMBinaryBroadcastCompute{func_subtract_cpu})
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_subtract"});

MXNET_OPERAOTR_REGISTER_BACKWARD_NP_BINARY_BROADCAST_USE_NONE(subtract);

#if MXNET_USE_CUDA
NNVM_REGISTER_OP(_npi_multiply)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastCompute{func_multiply_gpu});

NNVM_REGISTER_OP(_backward_npi_multiply)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastBackwardUseIn{func_backward_multiply_gpu});

NNVM_REGISTER_OP(_npi_add)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastCompute{func_add_gpu});

NNVM_REGISTER_OP(_backward_npi_add)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastBackwardUseNone{func_backward_add_gpu});

NNVM_REGISTER_OP(_npi_subtract)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastCompute{func_subtract_gpu});

NNVM_REGISTER_OP(_backward_npi_subtract)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastBackwardUseNone{func_backward_subtract_gpu});

#endif  // MXNET_USE_CUDA

static constexpr char func_multiply_scalar_cpu[] = "multiply_scalar_cpu";
static constexpr char func_multiply_scalar_gpu[] = "multiply_scalar_gpu";
static constexpr char func_backward_multiply_scalar_cpu[] = "backward_multiply_scalar_cpu";
static constexpr char func_backward_multiply_scalar_gpu[] = "backward_multiply_scalar_gpu";
static constexpr char func_add_scalar_cpu[] = "add_scalar_cpu";
static constexpr char func_add_scalar_gpu[] = "add_scalar_gpu";
static constexpr char func_backward_add_scalar_cpu[] = "backward_add_scalar_cpu";
static constexpr char func_backward_add_scalar_gpu[] = "backward_add_scalar_gpu";
static constexpr char func_subtract_scalar_cpu[] = "subtract_scalar_cpu";
static constexpr char func_subtract_scalar_gpu[] = "subtract_scalar_gpu";
static constexpr char func_backward_subtract_scalar_cpu[] = "backward_subtract_scalar_cpu";
static constexpr char func_backward_subtract_scalar_gpu[] = "backward_subtract_scalar_gpu";
static constexpr char func_rsubtract_scalar_cpu[] = "rsubtract_scalar_cpu";
static constexpr char func_rsubtract_scalar_gpu[] = "rsubtract_scalar_gpu";
static constexpr char func_backward_rsubtract_scalar_cpu[] = "backward_rsubtract_scalar_cpu";
static constexpr char func_backward_rsubtract_scalar_gpu[] = "backward_rsubtract_scalar_gpu";


MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_multiply_scalar)
.set_attr<FCompute>("FCompute<cpu>", TVMBinaryBroadcastScalarCompute{func_multiply_scalar_cpu})
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_multiply_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_backward_npi_multiply_scalar)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>",
  TVMBinaryBroadcastScalarCompute{func_backward_multiply_scalar_cpu});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_add_scalar)
.set_attr<FCompute>("FCompute<cpu>", TVMBinaryBroadcastScalarCompute{func_add_scalar_cpu})
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_add_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_backward_npi_add_scalar)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>",
  TVMBinaryBroadcastScalarCompute{func_backward_add_scalar_cpu});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_subtract_scalar)
.set_attr<FCompute>("FCompute<cpu>", TVMBinaryBroadcastScalarCompute{func_subtract_scalar_cpu})
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_subtract_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_backward_npi_subtract_scalar)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>",
  TVMBinaryBroadcastScalarCompute{func_backward_subtract_scalar_cpu});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_rsubtract_scalar)
.set_attr<FCompute>("FCompute<cpu>", TVMBinaryBroadcastScalarCompute{func_rsubtract_scalar_cpu})
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_rsubtract_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_backward_npi_rsubtract_scalar)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>",
  TVMBinaryBroadcastScalarCompute{func_backward_rsubtract_scalar_cpu});

#if MXNET_USE_CUDA
NNVM_REGISTER_OP(_npi_multiply_scalar)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastScalarCompute{func_multiply_scalar_gpu});

NNVM_REGISTER_OP(_backward_npi_multiply_scalar)
.set_attr<FCompute>("FCompute<gpu>",
                    TVMBinaryBroadcastScalarCompute{func_backward_multiply_scalar_gpu});

NNVM_REGISTER_OP(_npi_add_scalar)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastScalarCompute{func_add_scalar_gpu});

NNVM_REGISTER_OP(_backward_npi_add_scalar)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastScalarCompute{func_backward_add_scalar_gpu});

NNVM_REGISTER_OP(_npi_subtract_scalar)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastScalarCompute{func_subtract_scalar_gpu});

NNVM_REGISTER_OP(_backward_npi_subtract_scalar)
.set_attr<FCompute>("FCompute<gpu>",
                    TVMBinaryBroadcastScalarCompute{func_backward_subtract_scalar_gpu});

NNVM_REGISTER_OP(_npi_rsubtract_scalar)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastScalarCompute{func_rsubtract_scalar_gpu});

NNVM_REGISTER_OP(_backward_npi_rsubtract_scalar)
.set_attr<FCompute>("FCompute<gpu>",
                    TVMBinaryBroadcastScalarCompute{func_backward_rsubtract_scalar_gpu});

#endif  // MXNET_USE_CUDA

#else

MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(_npi_add)
#ifndef _WIN32
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastComputeWithBool<cpu, op::mshadow_op::plus, op::mshadow_op::mixed_plus,
                                      op::mshadow_op::mixed_plus>)
#else
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastComputeWithBool<cpu, op::mshadow_op::plus>)
#endif
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_broadcast_add"});

MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(_npi_multiply)
#ifndef _WIN32
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastComputeWithBool<cpu, op::mshadow_op::mul, op::mshadow_op::mixed_mul,
                                      op::mshadow_op::mixed_mul>)
#else
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastComputeWithBool<cpu, op::mshadow_op::mul>)
#endif
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_mul"});

MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(_npi_subtract)
#ifndef _WIN32
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastCompute<cpu, op::mshadow_op::minus, op::mshadow_op::mixed_minus,
                              op::mshadow_op::mixed_rminus>)
#else
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastCompute<cpu, op::mshadow_op::minus>)
#endif
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_broadcast_sub"});

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

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_mod)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::mod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_mod"});

MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(_npi_power)
#ifndef _WIN32
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastComputeWithBool<cpu, op::mshadow_op::power, op::mshadow_op::mixed_power,
                                      op::mshadow_op::mixed_rpower>)
#else
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastComputeWithBool<cpu, op::mshadow_op::power>)
#endif
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

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_multiply_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, op::mshadow_op::mul>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_mul_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_subtract_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, op::mshadow_op::minus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_rsubtract_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::rminus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"negative"});

#endif  // MXNET_USE_TVM_OP

}  // namespace op
}  // namespace mxnet
