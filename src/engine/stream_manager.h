/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_ENGINE_STREAM_MANAGER_H_
#define MXNET_ENGINE_STREAM_MANAGER_H_

#include <dmlc/base.h>
#include <cstddef>
#include <array>
#include <mutex>
#include "mxnet/base.h"
#include "mxnet/context.h"
#include "../common/cuda_utils.h"

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
  ~StreamManager();
  RunContext GetRunContext(Context const& ctx);

 private:
  std::mutex m_;
#if MXNET_USE_CUDA
  std::array<std::array<mshadow::Stream<gpu>*, kStreams>, kNumGpus>
      gpu_streams_;
  std::array<int, kNumGpus> gpu_cnt_;
#endif  // MXNET_USE_CUDA
  DISALLOW_COPY_AND_ASSIGN(StreamManager);
};  // class StreamManager

template <std::size_t kNumGpus, std::size_t kStreams>
RunContext StreamManager<kNumGpus, kStreams>::GetRuncontext(
    Context const& ctx) {
  switch (ctx.dev_mask) {
    case cpu::kDevMask:
      return {nullptr};
    case gpu::kDevMask: {
#if MXNET_USE_CUDA
      std::size_t use_counter;
      CUDA_CALL(cudaSetDevice(ctx.dev_id));
      {
        std::lock_guard<std::mutex> lock{m_};
        auto&& counter = gpu_cnt_.at(dev_id);
        if (counter == -1) {
          for (auto&& i : gpu_streams_.at(ctx.dev_id)) {
            i = mshadow::NewStream<gpu>(true, false);
          }
        }
        counter = 0;
      }
      use_counter = counter;
      counter = (counter + 1) % kStreams;
    }
      return {gpu_streams_.at(ctx.dev_id).at(use_counter)};
#else   // MXNET_USE_CUDA
      LOG(FATAL) << "Please compile with CUDA enabled";
      return {nullptr};
#endif  // MXNET_USE_CUDA
  }
}

template <std::size_t kNumGpus, std::size_t kStreams>
StreamManager<kNumGpus, kStreams>::StreamManager() {
#if MXNET_USE_CUDA
  for (std::size_t i = 0; i < kNumGpus; ++i) {
    gpu_cnt_.at(i) = -1;
  }
#endif  // MXNET_USE_CUDA
}

template <std::size_t>
StreamManager::~StreamManager() {
#if MXNET_USE_CUDA
  for (std::size_t i = 0; i < kNumGpus; ++i) {
    if (gpu_cnt_.at(i) != -1) {
      for (auto&& j : gpu_streams_.at(i)) {
        mshadow::DeleteStream<gpu>(j);
      }
    }
  }
#endif  // MXNET_USE_CUDA
}

}  // namespace engine

}  // namespace mxnet

#endif  // MXNET_ENGINE_STREAM_MANAGER_H_
