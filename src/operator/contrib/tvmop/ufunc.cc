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
static constexpr int max_dim = 5;

TBlob padding(const TBlob& tblob, const int max_dim) {
  TShape tshape(max_dim, 1);
  int ndim = tblob.shape_.ndim();
  for (int i = max_dim - ndim; i < max_dim; ++i) {
    tshape[i] = tblob.size(i - max_dim + ndim);
  }
  return tblob.reshape(tshape);
}

template<const char* func>
void TVMBinaryCompute(const nnvm::NodeAttrs& attrs,
                      const mxnet::OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  TBlob idata[2], odata;
  for (int k = 0; k < 2; ++k) {
    idata[k] = padding(inputs[k], max_dim);
  }
  odata = padding(outputs[0], max_dim);
  tvm::runtime::TVMOpModule::Get()->Call(func, ctx, {idata[0], idata[1], odata});
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
    // dispatch by backward
    std::vector<int> ov, iv;
    TBlob ograd = padding(inputs[0], ndim), igrad = padding(outputs[k], ndim);
    int flag;
    if (ograd.size(0) != igrad.size(0)) {
      flag = 1;
    } else {
      flag = 0;
    }
    for (int i = 0; i < ndim; ++i) {
      if (i == 0 || (ograd.size(i) != igrad.size(i)) != (ograd.size(i - 1) != igrad.size(i - 1))) {
        ov.push_back(ograd.size(i));
      } else {
        ov.back() *= ograd.size(i);
      }
    }
    for (int i = ov.size(); i < max_dim; ++i) {
      ov.push_back(1);
    }
    for (uint32_t i = flag; i < ov.size(); i += 2) {
      iv.push_back(ov[i]);
    }
    TShape oshape(ov.begin(), ov.end()), ishape(iv.begin(), iv.end());
    TBlob ograd_tvm(ograd.reshape(oshape));
    TBlob igrad_tvm(igrad.reshape(ishape));
    std::string funcname = std::string(func) + "reduce1st_" + std::to_string(flag);
    // dispatch by req
    funcname += "req_";
    MXNET_ASSIGN_REQ_SWITCH(req[k], req_type, {
      if (req_type == kWriteTo) {
        funcname += "kWriteTo";
      } else {
        funcname += "kAddTo";
      }
    })
    tvm::runtime::TVMOpModule::Get()->Call(funcname, ctx, {ograd_tvm, igrad_tvm, igrad_tvm});
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

inline bool DegandradOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  CHECK(in_attrs->at(0) == mshadow::kFloat64 ||
        in_attrs->at(0) == mshadow::kFloat32)
    << "Only support float32 and float64.";
  return out_attrs->at(0) != -1;
}

template<const char* func>
void TVMUnaryCompute(const nnvm::NodeAttrs& attrs,
                     const mxnet::OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (0 == inputs[0].shape_.Size()) {
    //  0-size
    return;
  }
  tvm::runtime::TVMOpModule::Get()->Call(func, ctx, {inputs[0], outputs[0]});
}

template<const char* func>
void TVMUnaryBackwardComputeUseNone(const nnvm::NodeAttrs& attrs,
                                    const mxnet::OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (0 == inputs[0].shape_.Size()) {
    //  0-size
    return;
  }
  std::string funcname = func;
  funcname += "req_";
  MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
    if (req_type == kWriteTo) {
      funcname += "kWriteTo";
    } else {
      funcname += "kAddTo";
    }
    tvm::runtime::TVMOpModule::Get()->Call(funcname, ctx,
      {inputs[0], outputs[0], outputs[0]});
  })
}

static constexpr char func_deg2rad_cpu[] = "deg2rad";
static constexpr char func_deg2rad_gpu[] = "cuda_deg2rad";
static constexpr char func_backward_deg2rad_cpu[] = "backward_deg2rad";
static constexpr char func_backward_deg2rad_gpu[] = "cuda_backward_deg2rad";

NNVM_REGISTER_OP(_npi_deg2rad)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
[](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data"};
})
.set_attr<mxnet::FInferShape>("FInferShape", mxnet::op::ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", mxnet::op::DegandradOpType)
#if MXNET_USE_CUDA
.set_attr<mxnet::FCompute>("FCompute<gpu>", mxnet::op::TVMUnaryCompute<func_deg2rad_gpu>)
#endif  // MXNET_USE_CUDA
.set_attr<mxnet::FCompute>("FCompute<cpu>", mxnet::op::TVMUnaryCompute<func_deg2rad_cpu>)
.add_argument("data", "NDArray-or-Symbol", "the input")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_deg2rad"});

NNVM_REGISTER_OP(_backward_npi_deg2rad)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
#if MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<gpu>",
                    mxnet::op::TVMUnaryBackwardComputeUseNone<func_backward_deg2rad_gpu>)
#endif  // MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<cpu>",
                    mxnet::op::TVMUnaryBackwardComputeUseNone<func_backward_deg2rad_cpu>);

static constexpr char func_rad2deg_cpu[] = "rad2deg";
static constexpr char func_rad2deg_gpu[] = "cuda_rad2deg";
static constexpr char func_backward_rad2deg_cpu[] = "backward_rad2deg";
static constexpr char func_backward_rad2deg_gpu[] = "cuda_backward_rad2deg";

NNVM_REGISTER_OP(_npi_rad2deg)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
[](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data"};
})
.set_attr<mxnet::FInferShape>("FInferShape", mxnet::op::ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", mxnet::op::DegandradOpType)
#if MXNET_USE_CUDA
.set_attr<mxnet::FCompute>("FCompute<gpu>", mxnet::op::TVMUnaryCompute<func_rad2deg_gpu>)
#endif  // MXNET_USE_CUDA
.set_attr<mxnet::FCompute>("FCompute<cpu>", mxnet::op::TVMUnaryCompute<func_rad2deg_cpu>)
.add_argument("data", "NDArray-or-Symbol", "the input")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_rad2deg"});

NNVM_REGISTER_OP(_backward_npi_rad2deg)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
#if MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<gpu>",
                    mxnet::op::TVMUnaryBackwardComputeUseNone<func_backward_rad2deg_gpu>)
#endif  // MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<cpu>",
                    mxnet::op::TVMUnaryBackwardComputeUseNone<func_backward_rad2deg_cpu>);
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_TVM_OP
