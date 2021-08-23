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
 * \file threaded_engine_pooled.cc
 * \brief Pooled threaded engine
 * \author Yutian Li
 */
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/concurrency.h>
#include <cassert>
#include <memory>
#include <utility>
#include "./threaded_engine.h"
#include "./thread_pool.h"
#include "./stream_manager.h"
#if MXNET_USE_CUDA
#include "../common/cuda/utils.h"
#endif

namespace mxnet {
namespace engine {
/*!
 * \brief ThreadedEngine using global thread pool across all devices.
 * The policy of this Engine:
 *  - Execute Async operation immediately if pushed from Pusher.
 *  - Use a common thread pool for normal operations on all devices.
 *  - Use special thread pool for copy operations.
 */
class ThreadedEnginePooled : public ThreadedEngine {
 public:
  ThreadedEnginePooled() {
    this->Start();
  }

  ~ThreadedEnginePooled() noexcept(false) override {
    StopNoWait();
  }

  void StopNoWait() {
    streams_->Finalize();
    task_queue_->SignalForKill();
    io_task_queue_->SignalForKill();
    task_queue_ = nullptr;
    io_task_queue_ = nullptr;
    thread_pool_ = nullptr;
    io_thread_pool_ = nullptr;
    streams_ = nullptr;
  }

  void Stop() override {
    WaitForAll();
    StopNoWait();
  }

  void Start() override {
    streams_ = std::make_unique<StreamManager<kMaxNumGpus, kNumStreamsPerGpu>>();
    task_queue_.reset(new dmlc::ConcurrentBlockingQueue<OprBlock*>());
    io_task_queue_.reset(new dmlc::ConcurrentBlockingQueue<OprBlock*>());
    thread_pool_ = std::make_unique<ThreadPool>(kNumWorkingThreads,
                                      [this](std::shared_ptr<dmlc::ManualEvent> ready_event) {
                                        ThreadWorker(task_queue_, ready_event); },
                                      true);
    io_thread_pool_ = std::make_unique<ThreadPool>(1,
                                         [this](std::shared_ptr<dmlc::ManualEvent> ready_event) {
                                           ThreadWorker(io_task_queue_, ready_event); },
                                         true);
  }

 protected:
  void PushToExecute(OprBlock *opr_block, bool pusher_thread) override {
    if (opr_block->opr->prop == FnProperty::kAsync && pusher_thread) {
      DoExecute(opr_block);
    } else {
      DoPushToQueue(opr_block);
    }
  }

 private:
  /*! \brief Concurrency for thread pool */
  static constexpr std::size_t kNumWorkingThreads = 16;
  /*! \brief Maximum number of GPUs */
  static constexpr std::size_t kMaxNumGpus = 16;
  /*!\brief number of streams allocated for each GPU */
  static constexpr std::size_t kNumStreamsPerGpu = 16;
  /*!
   * \brief Streams.
   */
  std::unique_ptr<StreamManager<kMaxNumGpus, kNumStreamsPerGpu>> streams_;
  /*!
   * \brief Task queues.
   */
  std::shared_ptr<dmlc::ConcurrentBlockingQueue<OprBlock*>> task_queue_;
  std::shared_ptr<dmlc::ConcurrentBlockingQueue<OprBlock*>> io_task_queue_;
  /*!
   * \brief Thread pools.
   */
  std::unique_ptr<ThreadPool> thread_pool_;
  std::unique_ptr<ThreadPool> io_thread_pool_;
  /*!
   * \brief Worker.
   * \param task_queue Queue to work on.
   *
   * The method to pass to thread pool to parallelize.
   */
  void ThreadWorker(std::shared_ptr<dmlc::ConcurrentBlockingQueue<OprBlock*>> task_queue,
                    const std::shared_ptr<dmlc::ManualEvent>& ready_event) {
    OprBlock* opr_block;
    ready_event->signal();
    while (task_queue->Pop(&opr_block)) {
      DoExecute(opr_block);
    }
  }
  /*!
   * \brief Execute an operation.
   * \param opr_block The operator block.
   */
  void DoExecute(OprBlock* opr_block) {
#if MXNET_USE_CUDA
    mxnet::common::cuda::DeviceStore device_store(-1, false);
#endif
    assert(opr_block->wait.load() == 0);
    if (opr_block->ctx.dev_mask() == gpu::kDevMask) {
      #if MXNET_USE_CUDA
      device_store.SetDevice(opr_block->ctx.dev_id);
      #else   // MXNET_USE_CUDA
      LOG(FATAL) << "Please compile with CUDA enabled";
      #endif  // MXNET_USE_CUDA
    }
    bool is_copy = (opr_block->opr->prop == FnProperty::kCopyFromGPU ||
                    opr_block->opr->prop == FnProperty::kCopyToGPU);
    auto&& rctx = is_copy
        ? streams_->GetIORunContext(opr_block->ctx)
        : streams_->GetRunContext(opr_block->ctx);
    this->ExecuteOprBlock(rctx, opr_block);
  }
  /*!
   * \brief Push the operation to the queue.
   * \param opr_block The operator block.
   */
  void DoPushToQueue(OprBlock* opr_block) {
    switch (opr_block->opr->prop) {
      case FnProperty::kCopyFromGPU:
      case FnProperty::kCopyToGPU: {
        io_task_queue_->Push(opr_block);
        break;
      }
      default: {
        task_queue_->Push(opr_block);
        break;
      }
    }
  }
};

Engine *CreateThreadedEnginePooled() {
  return new ThreadedEnginePooled();
}
}  // namespace engine
}  // namespace mxnet
