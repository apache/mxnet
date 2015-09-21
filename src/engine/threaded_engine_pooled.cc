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
#include "./threaded_engine.h"
#include "./thread_pool.h"
#include "./stream_manager.h"

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
  ThreadedEnginePooled() :
      thread_pool_(kNumWorkingThreads, [this]() { ThreadWorker(&task_queue_); }),
      io_thread_pool_(1, [this]() { ThreadWorker(&io_task_queue_); }) {}

  ~ThreadedEnginePooled() noexcept(false) {
    streams_.Finalize();
    task_queue_.SignalForKill();
    io_task_queue_.SignalForKill();
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
  StreamManager<kMaxNumGpus, kNumStreamsPerGpu> streams_;
  /*!
   * \brief Task queues.
   */
  dmlc::ConcurrentBlockingQueue<OprBlock*> task_queue_;
  dmlc::ConcurrentBlockingQueue<OprBlock*> io_task_queue_;
  /*!
   * \brief Thread pools.
   */
  ThreadPool thread_pool_;
  ThreadPool io_thread_pool_;
  /*!
   * \brief Worker.
   * \param task_queue Queue to work on.
   *
   * The method to pass to thread pool to parallelize.
   */
  void ThreadWorker(dmlc::ConcurrentBlockingQueue<OprBlock*>* task_queue) {
    OprBlock* opr_block;
    while (task_queue->Pop(&opr_block)) {
      DoExecute(opr_block);
    }
  }
  /*!
   * \brief Execute an operation.
   * \param opr_block The operator block.
   */
  void DoExecute(OprBlock* opr_block) {
    assert(opr_block->wait.load() == 0);
    if (opr_block->ctx.dev_mask() == gpu::kDevMask) {
      #if MXNET_USE_CUDA
      CUDA_CALL(cudaSetDevice(opr_block->ctx.dev_id));
      #else   // MXNET_USE_CUDA
      LOG(FATAL) << "Please compile with CUDA enabled";
      #endif  // MXNET_USE_CUDA
    }
    bool is_copy = (opr_block->opr->prop == FnProperty::kCopyFromGPU ||
                    opr_block->opr->prop == FnProperty::kCopyToGPU);
    auto&& rctx = is_copy
        ? streams_.GetIORunContext(opr_block->ctx)
        : streams_.GetRunContext(opr_block->ctx);
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
        io_task_queue_.Push(opr_block);
        break;
      }
      default: {
        task_queue_.Push(opr_block);
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
