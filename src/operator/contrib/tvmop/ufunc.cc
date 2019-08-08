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
 * \file ufunc.cc
 * \brief
 * \author Yizhi Liu
 */
#ifdef MXNET_USE_TVM_OP
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <mxnet/base.h>
#include "../../tensor/elemwise_binary_broadcast_op.h"
#include "../../tvmop/op_module.h"
#include "../../tensor/elemwise_binary_op.h"

namespace mxnet {
namespace op {

static constexpr char func_vadd_cpu[] = "vadd";
static constexpr char func_vadd_gpu[] = "cuda_vadd";

static constexpr char func_fmax_cpu[] = "fmax";
static constexpr char func_fmax_gpu[] = "cuda_fmax";

static constexpr char func_fmin_cpu[] = "fmin";
static constexpr char func_fmin_gpu[] = "cuda_fmin";

template<const char* func>
void TVMBroadcastCompute(const nnvm::NodeAttrs& attrs,
                         const mxnet::OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  tvm::runtime::TVMOpModule::Get()->Call(func, ctx, {inputs[0], inputs[1], outputs[0]});
}

NNVM_REGISTER_OP(_contrib_tvm_vadd)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .add_argument("a", "NDArray-or-Symbol", "first input")
    .add_argument("b", "NDArray-or-Symbol", "second input")
    .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
    .set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
#if MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<gpu>", mxnet::op::TVMBroadcastCompute<func_vadd_gpu>)
#endif  // MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<cpu>", mxnet::op::TVMBroadcastCompute<func_vadd_cpu>);

NNVM_REGISTER_OP(_contrib_tvm_fmax)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .add_argument("a", "NDArray-or-Symbol", "first input")
    .add_argument("b", "NDArray-or-Symbol", "second input")
    .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
    .set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
#if MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<gpu>", mxnet::op::TVMBroadcastCompute<func_fmax_gpu>)
#endif  // MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<cpu>", mxnet::op::TVMBroadcastCompute<func_fmax_cpu>);

NNVM_REGISTER_OP(_contrib_tvm_fmin)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .add_argument("a", "NDArray-or-Symbol", "first input")
    .add_argument("b", "NDArray-or-Symbol", "second input")
    .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
    .set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
#if MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<gpu>", mxnet::op::TVMBroadcastCompute<func_fmin_gpu>)
#endif  // MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<cpu>", mxnet::op::TVMBroadcastCompute<func_fmin_cpu>);

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_TVM_OP
