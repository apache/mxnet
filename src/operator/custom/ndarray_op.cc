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
 * Copyright (c) 2015 by Contributors
 * \file ndarray_op.cc
 * \brief
 * \author Junyuan Xie
*/
#include "./ndarray_op-inl.h"
#include <mxnet/base.h>
#include <mxnet/ndarray.h>

namespace mxnet {
namespace op {
template<>
Context NDArrayOp<cpu>::get_ctx() {
  return Context::CPU();
}

template<>
Operator *CreateOp<cpu>(NDArrayOpParam param) {
  return new NDArrayOp<cpu>(param);
}

#if MXNET_USE_CUDA
template<>
Context NDArrayOp<gpu>::get_ctx() {
  int dev_id;
  CHECK_EQ(cudaGetDevice(&dev_id), cudaSuccess);
  return Context::GPU(dev_id);
}

template<>
Operator* CreateOp<gpu>(NDArrayOpParam param) {
  return new NDArrayOp<gpu>(param);
}
#endif  // MXNET_USE_CUDA

template<typename xpu>
void NDArrayOp<xpu>::Forward(const OpContext &ctx,
                   const std::vector<TBlob> &in_data,
                   const std::vector<OpReqType> &req,
                   const std::vector<TBlob> &out_data,
                   const std::vector<TBlob> &aux_args) {
  using namespace mshadow;
  Context ndctx = get_ctx();
  std::vector<void*> ptrs;
  std::vector<Engine::VarHandle> ndvar;
  std::vector<int> tags;
  for (auto& i : req) CHECK_NE(i, kAddTo);

  for (auto& blob : in_data) {
    ptrs.push_back(reinterpret_cast<void*>(new NDArray(blob, ndctx.dev_id)));
    tags.push_back(0);
  }
  for (auto& blob : out_data) {
    NDArray* nd = new NDArray(blob, ndctx.dev_id);
    ptrs.push_back(reinterpret_cast<void*>(nd));
    ndvar.push_back(nd->var());
    tags.push_back(1);
  }
  std::sort(ndvar.begin(), ndvar.end());
  ndvar.resize(std::unique(ndvar.begin(), ndvar.end()) - ndvar.begin());

  std::vector<NDArray> ndcpy;
  for (auto& i : ptrs) {
    ndcpy.push_back(*reinterpret_cast<NDArray*>(i));
  }

  CHECK(param_.pinfo->forward(ptrs.size(), ptrs.data(), tags.data(), param_.pinfo->p_forward));
  Engine::Get()->PushAsync(
      [ndcpy, ctx](RunContext rctx, Engine::CallbackOnComplete on_complete) {
        ctx.async_on_complete();
        on_complete();
      }, ndctx, ndvar, {}, FnProperty::kNormal, 0, PROFILER_MESSAGE("NDArrayOpForward"));
}

template<typename xpu>
void NDArrayOp<xpu>::Backward(const OpContext &ctx,
                    const std::vector<TBlob> &out_grad,
                    const std::vector<TBlob> &in_data,
                    const std::vector<TBlob> &out_data,
                    const std::vector<OpReqType> &req,
                    const std::vector<TBlob> &in_grad,
                    const std::vector<TBlob> &aux_args) {
  using namespace mshadow;
  Context ndctx = get_ctx();
  std::vector<void*> ptrs;
  std::vector<Engine::VarHandle> ndvar;
  std::vector<int> tags;
  for (auto& i : req) CHECK_NE(i, kAddTo);

  for (auto& blob : in_data) {
    ptrs.push_back(reinterpret_cast<void*>(new NDArray(blob, ndctx.dev_id)));
    tags.push_back(0);
  }
  for (auto& blob : out_data) {
    ptrs.push_back(reinterpret_cast<void*>(new NDArray(blob, ndctx.dev_id)));
    tags.push_back(1);
  }
  for (auto& blob : in_grad) {
    NDArray* nd = new NDArray(blob, ndctx.dev_id);
    ptrs.push_back(reinterpret_cast<void*>(nd));
    ndvar.push_back(nd->var());
    tags.push_back(2);
  }
  std::sort(ndvar.begin(), ndvar.end());
  ndvar.resize(std::unique(ndvar.begin(), ndvar.end()) - ndvar.begin());
  for (auto& blob : out_grad) {
    ptrs.push_back(reinterpret_cast<void*>(new NDArray(blob, ndctx.dev_id)));
    tags.push_back(3);
  }

  std::vector<NDArray> ndcpy;
  for (auto& i : ptrs) {
    ndcpy.push_back(*reinterpret_cast<NDArray*>(i));
  }

  CHECK(param_.pinfo->backward(ptrs.size(), ptrs.data(), tags.data(), param_.pinfo->p_backward));
  Engine::Get()->PushAsync(
      [ndcpy, ctx](RunContext rctx, Engine::CallbackOnComplete on_complete){
        ctx.async_on_complete();
        on_complete();
      }, ndctx, ndvar, {}, FnProperty::kNormal, 0, PROFILER_MESSAGE("NDArrayOpBackward"));
}

Operator* NDArrayOpProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(NDArrayOpParam);

MXNET_REGISTER_OP_PROPERTY(_NDArray, NDArrayOpProp)
.describe("Stub for implementing an operator implemented in native frontend language with ndarray.")
.add_argument("data", "NDArray-or-Symbol[]", "Input data for the custom operator.")
.add_arguments(NDArrayOpParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
