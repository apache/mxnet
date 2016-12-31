/*!
 * Copyright (c) 2015 by Contributors
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
    gpu_copy_nthreads_ = dmlc::GetEnv("MXNET_GPU_COPY_NTHREADS", 1);
    cpu_worker_nthreads_ = dmlc::GetEnv("MXNET_CPU_WORKER_NTHREADS", 1);
    // create CPU task
    int cpu_priority_nthreads = dmlc::GetEnv("MXNET_CPU_PRIORITY_NTHREADS", 4);
    cpu_priority_worker_.reset(new ThreadWorkerBlock<kPriorityQueue>());
    cpu_priority_worker_->pool.reset(new ThreadPool(
        cpu_priority_nthreads, [this] {
          this->CPUWorker(cpu_priority_worker_.get());
        }));
    // GPU tasks will be created lazily
  }
  ~ThreadedEnginePerDevice() noexcept(false) {
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
      RunContext run_ctx;
      run_ctx.stream = nullptr;
      this->ExecuteOprBlock(run_ctx, opr_block);
    } else {
      if (ctx.dev_mask() == cpu::kDevMask) {
        if (opr_block->opr->prop == FnProperty::kCPUPrioritized) {
          cpu_priority_worker_->task_queue.Push(opr_block, opr_block->priority);
        } else {
          int dev_id = ctx.dev_id;
          int nthread = cpu_worker_nthreads_;
          cpu_normal_workers_.Get(dev_id, [this, dev_id, nthread]() {
              auto blk = new ThreadWorkerBlock<kWorkerQueue>();
              blk->pool.reset(new ThreadPool(nthread, [this, blk] () {
                    this->CPUWorker(blk);
                  }));
              return blk;
            })->task_queue.Push(opr_block, opr_block->priority);
        }
      } else {
        CHECK_EQ(ctx.dev_mask(), gpu::kDevMask);
        // GPU execution.
        FnProperty prop = opr_block->opr->prop;
        bool is_copy = (prop == FnProperty::kCopyFromGPU ||
                        prop == FnProperty::kCopyToGPU);
        int nthread = gpu_worker_nthreads_;
        int dev_id = ctx.dev_id;
        if (is_copy) {
          gpu_copy_workers_.Get(dev_id, [this, dev_id, is_copy, nthread]() {
              auto blk = new ThreadWorkerBlock<kCopyQueue>();
              blk->pool.reset(new ThreadPool(nthread, [this, dev_id, is_copy, blk] () {
                    this->GPUWorker(dev_id, is_copy, blk);
                  }));
              return blk;
            })->task_queue.Push(opr_block, opr_block->priority);
        } else {
          gpu_normal_workers_.Get(dev_id, [this, dev_id, is_copy, nthread]() {
              auto blk = new ThreadWorkerBlock<kWorkerQueue>();
              blk->pool.reset(new ThreadPool(nthread, [this, dev_id, is_copy, blk] () {
                    this->GPUWorker(dev_id, is_copy, blk);
                  }));
              return blk;
            })->task_queue.Push(opr_block, opr_block->priority);
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
    // destructor
    ~ThreadWorkerBlock() noexcept(false) {
      task_queue.SignalForKill();
    }
  };
  /*! \brief number of concurrent thread cpu worker uses */
  int cpu_worker_nthreads_;
  /*! \brief number of concurrent thread each gpu worker uses */
  int gpu_worker_nthreads_;
  /*! \brief number of concurrent thread each gpu copy worker uses */
  int gpu_copy_nthreads_;
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
  inline void GPUWorker(int dev_id,
                        bool is_copy_worker,
                        ThreadWorkerBlock<type> *block) {
    #if MXNET_USE_CUDA
    // allocate stream
    mshadow::SetDevice<gpu>(dev_id);
    RunContext run_ctx;
    mshadow::Stream<gpu> *stream;
    if (is_copy_worker) {
      stream = mshadow::NewStream<gpu>(false, false);
    } else {
      stream = mshadow::NewStream<gpu>(true, MXNET_USE_CUDNN != 0);
    }
    run_ctx.stream = stream;
    // execute task
    OprBlock* opr_block;
    auto* task_queue = &(block->task_queue);
    while (task_queue->Pop(&opr_block)) {
      this->ExecuteOprBlock(run_ctx, opr_block);
    }
    // Catch exception for CUDA driver shutdown
    MSHADOW_CATCH_ERROR(mshadow::DeleteStream<gpu>(stream));
    #endif
  }
  /*!
   * \brief CPU worker that performs operations on CPU.
   * \param block The task block of the worker.
   */
  template<dmlc::ConcurrentQueueType type>
  inline void CPUWorker(ThreadWorkerBlock<type> *block) {
    auto* task_queue = &(block->task_queue);
    RunContext run_ctx;
    run_ctx.stream = nullptr;
    // execute task
    OprBlock* opr_block;
    while (task_queue->Pop(&opr_block)) {
      this->ExecuteOprBlock(run_ctx, opr_block);
    }
  }
};

Engine *CreateThreadedEnginePerDevice() {
  return new ThreadedEnginePerDevice();
}
}  // namespace engine
}  // namespace mxnet
