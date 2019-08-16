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
#include <string>
#include "../../tensor/elemwise_binary_broadcast_op.h"
#include "../../tvmop/op_module.h"
#include "../../tensor/elemwise_binary_op.h"

namespace mxnet {
namespace op {

static constexpr char func_vadd_cpu[] = "vadd";
static constexpr char func_vadd_gpu[] = "cuda_vadd";
static constexpr char func_bakcward_vadd_cpu[] = "backward_vadd";
static constexpr char func_bakcward_vadd_gpu[] = "cuda_backward_vadd";

template<const char* func>
void TVMBinaryCompute(const nnvm::NodeAttrs& attrs,
                      const mxnet::OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  tvm::runtime::TVMOpModule::Get()->Call(func, ctx, {inputs[0], inputs[1], outputs[0]});
}

template<const char* func>
void TVMBinaryBackwardComputeUseNone(const nnvm::NodeAttrs& attrs,
                                     const mxnet::OpContext& ctx,
                                     const std::vector<TBlob>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U);
  int ndim = inputs[0].shape_.ndim();
  for (int k = 0; k < 2; ++k) {
    std::vector<int> ov, iv;
    const TBlob& ograd = inputs[0], igrad = outputs[k];
    bool flag = ograd.size(0) != igrad.size(0);
    for (int i = 0; i < ndim; ++i) {
      if (i == 0 || (ograd.size(i) != igrad.size(i)) != (ograd.size(i - 1) != igrad.size(i - 1))) {
        ov.push_back(ograd.size(i));
      } else {
        ov.back() *= ograd.size(i);
      }
    }
    for (int i = flag; i < ov.size(); i += 2) {
      iv.push_back(ov[i]);
    }
    TShape oshape(ov.begin(), ov.end()), ishape(iv.begin(), iv.end());
    TBlob ograd_tvm(ograd.reshape(oshape).dltensor());
    TBlob igrad_tvm(igrad.reshape(ishape).dltensor());
    std::string funcname = std::string(func) + "reduce1st_" + std::to_string(flag);
    tvm::runtime::TVMOpModule::Get()->Call(funcname, ctx, {ograd_tvm, igrad_tvm});
  }
}

NNVM_REGISTER_OP(_contrib_tvm_vadd)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .add_argument("a", "NDArray-or-Symbol", "first input")
    .add_argument("b", "NDArray-or-Symbol", "second input")
    .set_attr<nnvm::FListInputNames>("FListInputNames",
      [](const NodeAttrs& attrs) {
        return std::vector<std::string>{"a", "b"};
      })
    .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
    .set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
#if MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<gpu>", mxnet::op::TVMBinaryCompute<func_vadd_gpu>)
#endif  // MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<cpu>", mxnet::op::TVMBinaryCompute<func_vadd_cpu>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_contrib_tvm_vadd"});

NNVM_REGISTER_OP(_backward_contrib_tvm_vadd)
    .set_num_inputs(1)
    .set_num_outputs(2)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
#if MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<gpu>",
                               mxnet::op::TVMBinaryBackwardComputeUseNone<func_bakcward_vadd_gpu>)
#endif  // MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<cpu>",
                               mxnet::op::TVMBinaryBackwardComputeUseNone<func_bakcward_vadd_cpu>);

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_TVM_OP
