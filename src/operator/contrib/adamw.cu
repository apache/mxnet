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
 *  Copyright (c) 2018 by Contributors
 * \file adamw.cu
 * \brief Optimizer operators
 * \author Haibin Lin
 */
#include "./adamw-inl.h"

namespace mxnet {
namespace op {

template<template <typename xpu> class F>
inline void MPUpdateGPU(const nnvm::NodeAttrs& attrs,
                        const OpContext &ctx,
                        const std::vector<TBlob> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &outputs) {
  // copy to cpu and check NaN value
  TBlob scale_blob = inputs[inputs.size() - 1];
  MSHADOW_REAL_TYPE_SWITCH(scale_blob.type_flag_, DType, {
    DType scale = 0;
    CUDA_CALL(cudaMemcpy(&scale, scale_blob.dptr<DType>(), sizeof(DType),
       cudaMemcpyDeviceToHost));
    float scalef = static_cast<float>(scale);
    if (!std::isfinite(scalef) || scalef == 0) return;
    std::vector<TBlob> inputs_wo_scale;
    size_t num_in = inputs.size();
    inputs_wo_scale.reserve(num_in - 1);
    for (size_t i = 0; i < num_in - 1; i++) inputs_wo_scale.emplace_back(inputs[i]);
    F<gpu>::Forward(attrs, ctx, inputs_wo_scale, req, outputs, scalef);
  });
}

NNVM_REGISTER_OP(_contrib_adamw_update)
.set_attr<FCompute>("FCompute<gpu>", MPUpdateGPU<AdamWUpdate>);

NNVM_REGISTER_OP(_contrib_mp_adamw_update)
.set_attr<FCompute>("FCompute<gpu>", MPUpdateGPU<MPAdamWUpdate>);

}  // namespace op
}  // namespace mxnet
