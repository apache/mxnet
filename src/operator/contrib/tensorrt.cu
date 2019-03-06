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
 * Copyright (c) 2018 by Contributors
 * \file trt.cu
 * \brief TensorRT GPU operation
 * \author Marek Kolodziej, Clement Fuji Tsang
*/

#if MXNET_USE_TENSORRT

#include "./tensorrt-inl.h"

namespace mxnet {
namespace op {

#define CHECK_CUDART(x) do { \
  cudaError_t res = (x); \
  if (res != cudaSuccess) { \
    fprintf(stderr, "CUDART: %s = %d (%s) at (%s:%d)\n", \
      #x, res, cudaGetErrorString(res), __FILE__, __LINE__); \
    exit(1); \
  } \
} while (0)

void TRTCompute(const OpStatePtr& state, const OpContext& ctx,
                     const std::vector<TBlob>& inputs, const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;

  Stream<gpu>* s = ctx.get_stream<gpu>();
  cudaStream_t cuda_s = Stream<gpu>::GetStream(s);
  const auto& param = state.get_state<TRTEngineParam>();
  std::vector<void*> bindings;
  bindings.reserve(param.binding_map.size());
  for (auto& p : param.binding_map) {
    if (p.second == nnvm_to_onnx::TypeIO::Inputs) {
      bindings.emplace_back(inputs[p.first].dptr_);
    } else {
      bindings.emplace_back(outputs[p.first].dptr_);
    }
  }

  const int batch_size = static_cast<int>(inputs[0].shape_[0]);
  param.trt_executor->enqueue(batch_size, bindings.data(), cuda_s, nullptr);
  CHECK_CUDART(cudaStreamSynchronize(cuda_s));
}

NNVM_REGISTER_OP(_trt_op)
.set_attr<FStatefulCompute>("FStatefulCompute<gpu>", TRTCompute);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_TENSORRT
