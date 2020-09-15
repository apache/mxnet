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
 * Copyright (c) 2019-2020 by Contributors
 * \file tensorrt.cu
 * \brief TensorRT GPU operation registration
 * \author Marek Kolodziej, Clement Fuji Tsang, Serge Panev
*/

#if MXNET_USE_TENSORRT

#include <string>
#include <unordered_map>

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
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  cudaStream_t cuda_s = Stream<gpu>::GetStream(ctx.get_stream<gpu>());
  auto& param = state.get_state<TRTEngineParam>();
  if (param.calibration_mode) {
    std::unordered_map<std::string, void*> input_ptr_map;
    for (auto it : param.input_name_to_idx) {
      input_ptr_map.emplace(it.first, inputs[it.second].dptr_);
    }
    param.calibrator->setBatch(input_ptr_map, cuda_s);
  }
  for (size_t i = 0; i < param.binding_order->size(); ++i) {
    auto& p = param.binding_order->at(i);
    if (p.second == true) {
      param.bindings->at(i) = inputs[p.first].dptr_;
    } else {
      param.bindings->at(i) = outputs[p.first].dptr_;
    }
  }
  param.trt_executor->enqueueV2(param.bindings->data(), cuda_s, nullptr);

  if (param.calibration_mode && param.calibrator->lastIter()) {
    param.calibrator->waitAndSetDone();
    // calibrator is fully calibrated, the calibration tables are ready
    cudaStreamSynchronize(cuda_s);
    // create the new engine
    auto int8_engine = param.future_int8_engine.get();
    LOG(INFO) << "[TensorRT op] Calibration done, setting inference engine to INT8.";
    param.ResetEngine(std::move(int8_engine),
                      /* calibration_mode=*/ false);
  }
}

NNVM_REGISTER_OP(_TensorRT)
.set_attr<FStatefulCompute>("FStatefulCompute<gpu>", TRTCompute)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_TENSORRT
