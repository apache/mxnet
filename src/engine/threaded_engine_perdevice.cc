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
 * \file threaded_engine_perdevice.cc
 * \brief ThreadedEngine that uses fix amount of thread for each device.
 */
#include <dmlc/base.h>
#include <dmlc/omp.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/concurrency.h>
#include "./threaded_engine.h"
#include "./thread_pool.h"
#include "../common/lazy_alloc_array.h"
#include "../common/utils.h"

namespace mxnet {
namespace engine {
/*!
 * \brief ThreadedEngine uses per device threads.
 * The policy of this Engine:
 *  - Execute Async operation immediately if pushed from Pusher.
 *  - Use fixed amount of threads for each device.
 *  - Use special threads for copy operations.
 *  - Each stream is allocated and bound to each of the thread.
 */
class ThreadedEnginePerDevice : public ThreadedEngine {
 public:
  static auto constexpr kFIFO = dmlc::ConcurrentQueueType::kFIFO;
  static auto constexpr kPriority = dmlc::ConcurrentQueueType::kPriority;
  static auto constexpr kCopyQueue = kPriority;
  static auto constexpr kPriorityQueue = kPriority;
  static auto constexpr kWorkerQueue = kFIFO;

  ThreadedEnginePerDevice() noexcept(false) {
    gpu_worker_nthreads_ = common::GetNumThreadPerGPU();
    cpu_worker_nthreads_ = dmlc::GetEnv("MXNET_CPU_WORKER_NTHREADS", 1);
    // create CPU task
    int cpu_priority_nthreads = dmlc::GetEnv("MXNET_CPU_PRIORITY_NTHREADS", 4);
    cpu_priority_worker_.reset(new ThreadWorkerBlock<kPriorityQueue>());
    cpu_priority_worker_->pool.reset(new ThreadPool(
        cpu_priority_nthreads, [this]() {
          this->CPUWorker(Context(), cpu_priority_worker_.get());
        }));
    // GPU tasks will be created lazily
  }
  ~ThreadedEnginePerDevice() noexcept(false) {
    SignalQueuesForKill();
    gpu_normal_workers_.Clear();
    gpu_copy_workers_.Clear();
    cpu_normal_workers_.Clear();
    cpu_priority_worker_.reset(nullptr);
  }

 protected:
  void PushToExecute(OprBlock *opr_block, bool pusher_thread) override {
    const Context& ctx = opr_block->ctx;
    if (opr_block->opr->prop == FnProperty::kAsync && pusher_thread) {
      if (ctx.dev_mask() == gpu::kDevMask) {
        #if MXNET_USE_CUDA
        MSHADOW_CATCH_ERROR(mshadow::SetDevice<gpu>(ctx.dev_id));
        #endif
      }
      this->ExecuteOprBlock(RunContext{ctx, nullptr}, opr_block);
    } else {
      if (ctx.dev_mask() == cpu::kDevMask) {
        if (opr_block->opr->prop == FnProperty::kCPUPrioritized) {
          cpu_priority_worker_->task_queue.Push(opr_block, opr_block->priority);
        } else {
          int dev_id = ctx.dev_id;
          int nthread = cpu_worker_nthreads_;
          auto ptr =
          cpu_normal_workers_.Get(dev_id, [this, ctx, nthread]() {
              auto blk = new ThreadWorkerBlock<kWorkerQueue>();
              blk->pool.reset(new ThreadPool(nthread, [this, ctx, blk] () {
                    this->CPUWorker(ctx, blk);
                  }));
              return blk;
            });
          if (ptr) {
            ptr->task_queue.Push(opr_block, opr_block->priority);
          }
        }
      } else {
        CHECK_EQ(ctx.dev_mask(), gpu::kDevMask);
        // GPU execution.
        FnProperty prop = opr_block->opr->prop;
        bool is_copy = (prop == FnProperty::kCopyFromGPU ||
                        prop == FnProperty::kCopyToGPU);
        int nthread = gpu_worker_nthreads_;
        if (is_copy) {
          auto ptr =
          gpu_copy_workers_.Get(ctx.dev_id, [this, ctx, is_copy, nthread]() {
              auto blk = new ThreadWorkerBlock<kCopyQueue>();
              blk->pool.reset(new ThreadPool(
                nthread,
                [this, ctx, is_copy, blk]
                  (std::shared_ptr<ThreadPool::SimpleEvent> ready_event) {
                    this->GPUWorker(ctx, is_copy, blk, ready_event);
                  }, true));
              return blk;
            });
          if (ptr) {
            ptr->task_queue.Push(opr_block, opr_block->priority);
          }
        } else {
          auto ptr = gpu_normal_workers_.Get(ctx.dev_id, [this, ctx, is_copy, nthread]() {
              auto blk = new ThreadWorkerBlock<kWorkerQueue>();
              blk->pool.reset(new ThreadPool(
                nthread,
                [this, ctx, is_copy, blk]
                  (std::shared_ptr<ThreadPool::SimpleEvent> ready_event) {
                    this->GPUWorker(ctx, is_copy, blk, ready_event);
                  }, true));
              return blk;
            });
          if (ptr) {
            ptr->task_queue.Push(opr_block, opr_block->priority);
          }
        }
      }
    }
  }

 private:
  // working unit for each of the task.
  template<dmlc::ConcurrentQueueType type>
  struct ThreadWorkerBlock {
    // task queue on this task
    dmlc::ConcurrentBlockingQueue<OprBlock*, type>  task_queue;
    // thread pool that works on this task
    std::unique_ptr<ThreadPool> pool;
    // constructor
    ThreadWorkerBlock() = default;
    // destructor
    ~ThreadWorkerBlock() noexcept(false) {}
  };

  /*! \brief number of concurrent thread cpu worker uses */
  int cpu_worker_nthreads_;
  /*! \brief number of concurrent thread each gpu worker uses */
  int gpu_worker_nthreads_;
  // cpu worker
  common::LazyAllocArray<ThreadWorkerBlock<kWorkerQueue> > cpu_normal_workers_;
  // cpu priority worker
  std::unique_ptr<ThreadWorkerBlock<kPriorityQueue> > cpu_priority_worker_;
  // workers doing normal works on GPU
  common::LazyAllocArray<ThreadWorkerBlock<kWorkerQueue> > gpu_normal_workers_;
  // workers doing copy works from/to GPU
  common::LazyAllocArray<ThreadWorkerBlock<kCopyQueue> > gpu_copy_workers_;
  /*!
   * \brief GPU worker that performs operations on a certain device.
   * \param dev_id The device id of the worker.
   * \param is_copy_worker whether the worker only do copy job
   * \param block The task block of the worker.
   */
  template<dmlc::ConcurrentQueueType type>
  inline void GPUWorker(Context ctx,
                        bool is_copy_worker,
                        ThreadWorkerBlock<type> *block,
                        std::shared_ptr<ThreadPool::SimpleEvent> ready_event) {
#if MXNET_USE_CUDA
    mshadow::Stream<gpu> *stream;
    do {
      ThreadPool::SimpleEvent::SetReadyOnDestroy setReady(ready_event);
      // allocate stream
      mshadow::SetDevice<gpu>(ctx.dev_id);
      if (is_copy_worker) {
        stream = mshadow::NewStream<gpu>(false, false, ctx.dev_id);
      } else {
        stream = mshadow::NewStream<gpu>(true, MXNET_USE_CUDNN != 0, ctx.dev_id);
      }
    } while (false);
    // execute task
    OprBlock* opr_block;
    RunContext run_ctx{ctx, stream};
    auto* task_queue = &(block->task_queue);
    while (task_queue->Pop(&opr_block)) {
      this->ExecuteOprBlock(run_ctx, opr_block);
    }
    // Catch exception for CUDA driver shutdown
    MSHADOW_CATCH_ERROR(mshadow::DeleteStream<gpu>(stream));
#else
    ready_event->signal();
#endif
  }
  /*!
   * \brief CPU worker that performs operations on CPU.
   * \param block The task block of the worker.
   */
  template<dmlc::ConcurrentQueueType type>
  inline void CPUWorker(Context ctx,
                        ThreadWorkerBlock<type> *block) {
    auto* task_queue = &(block->task_queue);
    RunContext run_ctx{ctx, nullptr};
    // execute task
    OprBlock* opr_block;
    while (task_queue->Pop(&opr_block)) {
      this->ExecuteOprBlock(run_ctx, opr_block);
    }
  }

/*! \brief Signal a single queue for shutdown */
  template<typename Object>
  static inline void SignalQueueForKill(common::LazyAllocArray<Object> *array) {
    array->ForEach([](size_t i, Object *block) {
      block->task_queue.SignalForKill();
    });
  }

  /*! Signal all queues for shutdown */
  void SignalQueuesForKill() {
    SignalQueueForKill(&gpu_normal_workers_);
    SignalQueueForKill(&gpu_copy_workers_);
    SignalQueueForKill(&cpu_normal_workers_);
    if (cpu_priority_worker_) {
      cpu_priority_worker_->task_queue.SignalForKill();
    }
  }
};

Engine *CreateThreadedEnginePerDevice() {
  return new ThreadedEnginePerDevice();
}
}  // namespace engine
}  // namespace mxnet
