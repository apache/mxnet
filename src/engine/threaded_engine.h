/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_ENGINE_THREADED_ENGINE_H_
#define MXNET_ENGINE_THREADED_ENGINE_H_

#include <dmlc/base.h>
#include <dmlc/concurrency.h>
#include <dmlc/logging.h>
#include <vector>
#include <functional>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include "engine_impl.h"
#include "thread_pool.h"
#include "stream_manager.h"
#include "../common/object_pool.h"

namespace mxnet {

namespace engine {

/*!
 * \brief Forward declarations.
 */
struct ThreadedOpr;

/*!
 * \brief Operation in the queue.
 */
struct OprBlock : public common::ObjectPoolAllocatable<OprBlock> {
#if ENGINE_DEBUG
  static std::atomic<std::size_t> counter;
  OprBlock() { LOG(INFO) << __func__ << " " << ++counter; }
  ~OprBlock() { LOG(INFO) << __func__ << " " << --counter; }
#endif  // ENGINE_DEBUG
  std::atomic<std::size_t> wait{0};
  ThreadedOpr* opr{nullptr};
  Context ctx;
};  // struct OprBlock

/*!
 * \brief Variable with version information.
 */
struct VersionedVarBlock
    : public common::ObjectPoolAllocatable<VersionedVarBlock> {
#if ENGINE_DEBUG
  static std::atomic<std::size_t> counter;
  VersionedVarBlock() { LOG(INFO) << __func__ << " " << ++counter; }
  ~VersionedVarBlock() { LOG(INFO) << __func__ << " " << --counter; }
#endif  // ENGINE_DEBUG
  VersionedVarBlock* next{nullptr};
  OprBlock* trigger{nullptr};
  bool write{false};
};  // struct VersionedVarBlock

/*!
 * \brief Variable implementation.
 */
class ThreadedVar final : public Var,
                          public common::ObjectPoolAllocatable<ThreadedVar> {
 public:
#if ENGINE_DEBUG
  static std::atomic<std::size_t> counter;
  ~ThreadedVar() { LOG(INFO) << __func__ << " " << --counter; }
#endif  // ENGINE_DEBUG
  explicit ThreadedVar(VersionedVarBlock* head);
  void AppendReadDependency(OprBlock* opr_block);
  void AppendWriteDependency(OprBlock* opr_block);
  template <typename Dispatcher>
  void CompleteReadDependency(Dispatcher dispatcher);
  template <typename Dispatcher>
  bool CompleteWriteDependency(Dispatcher dispatcher);
  void SetToDelete();

  static ThreadedVar* CastFromBase(Var* ptr);

 private:
  // TODO(hotpxl) change this to spinlock for faster runtime
  std::mutex m_;
  std::size_t num_pending_reads_{0};
  VersionedVarBlock* head_{nullptr};
  VersionedVarBlock* pending_write_{nullptr};
  /*!
   * If true, then there are no current or future processing of the chain.
   */
  bool ready_to_read_{true};
  /*!
   * If true, delete after operation completes.
   */
  bool to_delete_{false};
};  // struct ThreadedVar

/*!
 * \brief Operator implementation.
 */
struct ThreadedOpr final : public Opr,
                           public common::ObjectPoolAllocatable<ThreadedOpr> {
#if ENGINE_DEBUG
  static std::atomic<std::size_t> counter;
  ThreadedOpr() { LOG(INFO) << __func__ << " " << ++counter; }
  ~ThreadedOpr() { LOG(INFO) << __func__ << " " << --counter; }
#endif  // ENGINE_DEBUG
  Engine::AsyncFn fn;
  std::vector<ThreadedVar*> const_vars;
  std::vector<ThreadedVar*> mutable_vars;
  bool temporary{false};

  static ThreadedOpr* CastFromBase(Opr* ptr);
};  // struct ThreadedOpr

/*!
 * \brief Engine implementation.
 */
class ThreadedEngine final : public Engine {
 public:
  /*!
   * \brief Constructor and destructor.
   */
  ThreadedEngine();
  ~ThreadedEngine() noexcept(false);
  /*!
   * \brief Overriding methods.
   */
  ThreadedVar* NewVariable() override;
  ThreadedOpr* NewOperator(AsyncFn fn, std::vector<VarHandle> const& const_vars,
                           std::vector<VarHandle> const& mutable_vars) override;
  void DeleteOperator(OprHandle op) override;
  void Push(OprHandle op, Context exec_ctx) override;
  void Push(Fn exec_fun, Context exec_ctx,
            std::vector<VarHandle> const& const_vars,
            std::vector<VarHandle> const& mutable_vars) override;
  void PushAsync(AsyncFn exec_fun, Context exec_ctx,
                 std::vector<VarHandle> const& const_vars,
                 std::vector<VarHandle> const& mutable_vars) override;
  void DeleteVariable(Fn delete_fn, Context exec_ctx, VarHandle var) override;
  void WaitForVar(VarHandle var) override;
  void WaitForAll() override;
  /*!
   * \brief Callback on operation completion.
   *
   * On operation completion, this will trigger subsequent operations.
   */
  void OnComplete(ThreadedOpr* threaded_opr);
  /*!
   * \brief Worker.
   *
   * The method to pass to thread pool to parallelize.
   */
  void ThreadWorker();

 private:
  /*!
   * \brief Concurrency for thread pool.
   */
  static constexpr std::size_t kNumWorkingThreads = 16;
  /*!
   * \brief Constants for runtime context.
   */
  static constexpr std::size_t kMaxNumGpus = 16;
  static constexpr std::size_t kNumStreamsPerGpu = 16;
  /*!
   * \brief Number of pending operations.
   */
  std::atomic<std::size_t> pending_;
  /*!
   * \brief Notify waits for single or all variables.
   */
  std::mutex finished_m_;
  std::condition_variable finished_cv_;
  /*!
   * \brief Streams.
   */
  StreamManager<kMaxNumGpus, kNumStreamsPerGpu> streams_;
  /*!
   * \brief Task queue.
   */
  dmlc::ConcurrentBlockingQueue<OprBlock*> task_queue_;
  /*!
   * \brief Thread pool.
   */
  ThreadPool<kNumWorkingThreads> thread_pool_;
  /*!
   * \brief Disallow copy construction and assignment.
   */
  DISALLOW_COPY_AND_ASSIGN(ThreadedEngine);
};  // class ThreadedEngine

}  // namespace engine

}  // namespace mxnet

#endif  // MXNET_ENGINE_THREADED_ENGINE_H_
