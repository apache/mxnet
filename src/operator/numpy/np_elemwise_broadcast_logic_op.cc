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
 * \file np_elemwise_binary_logic_op.cc
 * \brief CPU Implementation of basic logic functions for elementwise numpy binary
 * broadcast operator.
 */

#include "./np_elemwise_broadcast_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

static constexpr char func_equal_cpu[] = "equal_cpu";
static constexpr char func_equal_gpu[] = "equal_gpu";
static constexpr char func_not_equal_cpu[] = "not_equal_cpu";
static constexpr char func_not_equal_gpu[] = "not_equal_gpu";
static constexpr char func_greater_cpu[] = "greater_cpu";
static constexpr char func_greater_gpu[] = "greater_gpu";
static constexpr char func_less_cpu[] = "less_cpu";
static constexpr char func_less_gpu[] = "less_gpu";
static constexpr char func_greater_equal_cpu[] = "greater_equal_cpu";
static constexpr char func_greater_equal_gpu[] = "greater_equal_gpu";
static constexpr char func_less_equal_cpu[] = "less_equal_cpu";
static constexpr char func_less_equal_gpu[] = "less_equal_gpu";

bool NumpyBinaryLogicOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (in_attrs->at(0) == -1 && in_attrs->at(1) == -1) return false;
  TYPE_ASSIGN_CHECK(*in_attrs, 0, in_attrs->at(1));
  TYPE_ASSIGN_CHECK(*in_attrs, 1, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kBool);
  return true;
}

#define MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(name)                                 \
  NNVM_REGISTER_OP(_npi_##name)                                                       \
  .set_num_inputs(2)                                                                  \
  .set_num_outputs(1)                                                                 \
  .set_attr<nnvm::FListInputNames>("FListInputNames",                                 \
  [](const NodeAttrs& attrs) {                                                        \
    return std::vector<std::string>{"lhs", "rhs"};                                    \
  })                                                                                  \
  .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)                  \
  .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryLogicOpType)                   \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                                   \
  [](const NodeAttrs& attrs) {                                                        \
    return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};                         \
  })                                                                                  \
  .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)                          \
  .add_argument("lhs", "NDArray-or-Symbol", "First input to the function")            \
  .add_argument("rhs", "NDArray-or-Symbol", "Second input to the function")

MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(not_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(greater);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(less);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(greater_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(less_equal);

#if MXNET_USE_TVM_OP

#define MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(name)                          \
  NNVM_REGISTER_OP(_npi_##name)                                                    \
  .set_attr<FCompute>("FCompute<cpu>", TVMBinaryBroadcastCompute{func_##name##_cpu})

#if MXNET_USE_CUDA

#define MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(name)                          \
  NNVM_REGISTER_OP(_npi_##name)                                                    \
  .set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastCompute{func_##name##_gpu})

MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(not_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(greater);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(less);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(greater_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(less_equal);

#endif  // MXNET_USE_CUDA

#else

#define MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(name)                                     \
  NNVM_REGISTER_OP(_npi_##name)                                                               \
  .set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastComputeLogic<cpu, mshadow_op::np_##name>)

#endif  // MXNET_USE_TVM_OP

MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(not_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(greater);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(less);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(greater_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(less_equal);

bool NumpyBinaryScalarLogicOpType(const nnvm::NodeAttrs& attrs,
                                  std::vector<int>* in_attrs,
                                  std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (in_attrs->at(0) == -1) return false;
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kBool);
  return true;
}

#define MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(name)                                \
  NNVM_REGISTER_OP(_npi_##name##_scalar)                                                    \
  .set_num_inputs(1)                                                                        \
  .set_num_outputs(1)                                                                       \
  .set_attr_parser([](NodeAttrs* attrs) {                                                   \
    attrs->parsed = std::stod(attrs->dict["scalar"]);                                       \
  })                                                                                        \
  .set_attr<nnvm::FListInputNames>("FListInputNames",                                       \
  [](const NodeAttrs& attrs) {                                                              \
    return std::vector<std::string>{"data"};                                                \
  })                                                                                        \
  .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)                         \
  .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryScalarLogicOpType)                   \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                                         \
  [](const NodeAttrs& attrs) {                                                              \
    return std::vector<std::pair<int, int> >{{0, 0}};                                       \
  })                                                                                        \
  .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)                                \
  .add_argument("data", "NDArray-or-Symbol", "First input to the function")                 \
  .add_argument("scalar", "float", "scalar input")

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(not_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(greater);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(less);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(greater_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(less_equal);

static constexpr char func_equal_scalar_cpu[] = "equal_scalar_cpu";
static constexpr char func_equal_scalar_gpu[] = "equal_scalar_gpu";
static constexpr char func_not_equal_scalar_cpu[] = "not_equal_scalar_cpu";
static constexpr char func_not_equal_scalar_gpu[] = "not_equal_scalar_gpu";
static constexpr char func_greater_scalar_cpu[] = "greater_scalar_cpu";
static constexpr char func_greater_scalar_gpu[] = "greater_scalar_gpu";
static constexpr char func_less_scalar_cpu[] = "less_scalar_cpu";
static constexpr char func_less_scalar_gpu[] = "less_scalar_gpu";
static constexpr char func_greater_equal_scalar_cpu[] = "greater_equal_scalar_cpu";
static constexpr char func_greater_equal_scalar_gpu[] = "greater_equal_scalar_gpu";
static constexpr char func_less_equal_scalar_cpu[] = "less_equal_scalar_cpu";
static constexpr char func_less_equal_scalar_gpu[] = "less_equal_scalar_gpu";

#if MXNET_USE_TVM_OP

#define MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(name)                                \
  NNVM_REGISTER_OP(_npi_##name##_scalar)                                                        \
  .set_attr<FCompute>("FCompute<cpu>", TVMBinaryBroadcastScalarCompute{func_##name##_scalar_cpu})

#if MXNET_USE_CUDA

#define MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(name)                                \
  NNVM_REGISTER_OP(_npi_##name##_scalar)                                                        \
  .set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastScalarCompute{func_##name##_scalar_gpu})

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(not_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(greater);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(less);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(greater_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(less_equal);

#endif  // MXNET_USE_CUDA

#else

#define MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(name)                               \
  NNVM_REGISTER_OP(_npi_##name##_scalar)                                                       \
  .set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::ComputeLogic<cpu, mshadow_op::np_##name>)

#endif  // MXNET_USE_TVM_OP

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(not_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(greater);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(less);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(greater_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(less_equal);

}  // namespace op
}  // namespace mxnet
