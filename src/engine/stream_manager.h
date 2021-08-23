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
 */
#ifndef MXNET_ENGINE_STREAM_MANAGER_H_
#define MXNET_ENGINE_STREAM_MANAGER_H_

#include <dmlc/base.h>
#include <mxnet/base.h>
#include <cstddef>
#include <array>
#include <string>
#include <mutex>
#include "../common/cuda/utils.h"

namespace mxnet {
namespace engine {

/*!
 * \brief Stream manager.
 *
 * Uses a basic round-robin algorithm to dispatch GPU streams. Returns default
 * context on CPU.
 */
template <std::size_t kNumGpus, std::size_t kStreams>
class StreamManager {
 public:
  StreamManager();
  ~StreamManager() {
    Finalize();
  }
  RunContext GetRunContext(Context const& ctx);
  RunContext GetIORunContext(Context const& ctx);
  void Finalize();
 private:
  std::mutex mutex_;
#if MXNET_USE_CUDA
  std::array<std::array<mshadow::Stream<gpu>*, kStreams>, kNumGpus>
      gpu_streams_;
  std::array<std::array<GPUAuxStream*, kStreams>, kNumGpus>
      gpu_aux_streams_;
  std::array<mshadow::Stream<gpu>*, kNumGpus> gpu_io_streams_;
  std::array<int, kNumGpus> gpu_cnt_;
#endif  // MXNET_USE_CUDA
  DISALLOW_COPY_AND_ASSIGN(StreamManager);
};  // class StreamManager

template <std::size_t kNumGpus, std::size_t kStreams>
RunContext StreamManager<kNumGpus, kStreams>::GetRunContext(
    Context const& ctx) {
  RunContext ret;
  switch (ctx.dev_mask()) {
    case cpu::kDevMask:
      ret = RunContext{ctx, nullptr, nullptr, false};
      break;
    case gpu::kDevMask: {
#if MXNET_USE_CUDA
      std::size_t use_counter;
      {
        std::lock_guard<std::mutex> lock{mutex_};
        auto&& counter = gpu_cnt_.at(ctx.dev_id);
        if (counter == -1) {
          mxnet::common::cuda::DeviceStore device_store(ctx.dev_id);
          for (auto&& primary_stream : gpu_streams_.at(ctx.dev_id)) {
            primary_stream = mshadow::NewStream<gpu>(true, MXNET_USE_CUDNN != 0, ctx.dev_id);
          }
          int idx = 0;
          for (auto&& aux_stream : gpu_aux_streams_.at(ctx.dev_id)) {
            auto primary_stream = gpu_streams_.at(ctx.dev_id).at(idx++);
            aux_stream = new GPUAuxStream(primary_stream);
          }
          counter = 0;
        }
        use_counter = counter;
        counter = (counter + 1) % kStreams;
      }
      ret = RunContext{ctx,
                       gpu_streams_.at(ctx.dev_id).at(use_counter),
                       gpu_aux_streams_.at(ctx.dev_id).at(use_counter),
                       false};
      break;
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif  // MXNET_USE_CUDA
    default:
      LOG(FATAL) << "Not Reached";
    }
  }
  return ret;
}

template <std::size_t kNumGpus, std::size_t kStreams>
RunContext StreamManager<kNumGpus, kStreams>::GetIORunContext(
    Context const& ctx) {
  RunContext ret;
  switch (ctx.dev_mask()) {
    case cpu::kDevMask:
      ret = RunContext{ctx, nullptr, nullptr, false};
      break;
    case gpu::kDevMask: {
#if MXNET_USE_CUDA
      {
        std::lock_guard<std::mutex> lock{mutex_};
        if (gpu_io_streams_.at(ctx.dev_id) == nullptr) {
          mxnet::common::cuda::DeviceStore device_store(ctx.dev_id);
          gpu_io_streams_.at(ctx.dev_id) = mshadow::NewStream<gpu>(false, false, ctx.dev_id);
        }
      }
      ret = RunContext{ctx, gpu_io_streams_.at(ctx.dev_id), nullptr, false};
      break;
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif  // MXNET_USE_CUDA
    default:
      LOG(FATAL) << "Not Reached";
    }
  }
  return ret;
}

template <std::size_t kNumGpus, std::size_t kStreams>
StreamManager<kNumGpus, kStreams>::StreamManager() {
#if MXNET_USE_CUDA
  for (std::size_t i = 0; i < kNumGpus; ++i) {
    gpu_cnt_.at(i) = -1;
  }
  for (auto&& i : gpu_io_streams_) {
    i = nullptr;
  }
#endif  // MXNET_USE_CUDA
}

template <std::size_t kNumGpus, std::size_t kStreams>
void StreamManager<kNumGpus, kStreams>::Finalize() {
#if MXNET_USE_CUDA
  for (std::size_t i = 0; i < kNumGpus; ++i) {
    if (gpu_cnt_.at(i) != -1) {
      for (auto&& primary_stream : gpu_streams_.at(i)) {
        // Catch exception for CUDA driver shutdown
        MSHADOW_CATCH_ERROR(mshadow::DeleteStream<gpu>(primary_stream));
      }
      for (auto&& aux_stream : gpu_aux_streams_.at(i)) {
        delete aux_stream;
      }
      gpu_cnt_.at(i) = -1;
    }
  }
#endif  // MXNET_USE_CUDA
}

}  // namespace engine
}  // namespace mxnet

#endif  // MXNET_ENGINE_STREAM_MANAGER_H_
