/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_DAG_ENGINE_SIMPLE_ENGINE_H_
#define MXNET_DAG_ENGINE_SIMPLE_ENGINE_H_

#include <dmlc/base.h>
#include <dmlc/concurrency.h>
#include <vector>
#include <functional>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include "mxnet/dag_engine.h"
#include "dag_engine_impl.h"
#include "thread_pool.h"

namespace mxnet {

namespace engine {

/*!
 * \brief Forward declarations.
 */
struct SimpleOpr;
struct OprBlock;

/*!
 * \brief Variable with version information.
 */
struct VersionedVarBlock {
  VersionedVarBlock* next{nullptr};
  VersionedVarBlock* join{nullptr};
  OprBlock* trigger{nullptr};
  dmlc::Spinlock lock;
  bool waiting{false};
};  // struct VersionedVarBlock

/*!
 * \brief Operation in the queue.
 */
struct OprBlock {
  std::function<void()> fn;
  std::atomic<std::size_t> wait{0};
};  // struct OprBlock

/*!
 * \brief Variable implementation.
 */
struct SimpleVar final : public Var {
  VersionedVarBlock* var{nullptr};

  static SimpleVar* CastFromBase(Var* ptr);
};  // struct SimpleVar

/*!
 * \brief Operator implementation.
 */
struct SimpleOpr final : public Opr {
  DAGEngine::AsyncFn fn;
  std::vector<SimpleVar*> use_vars;
  std::vector<SimpleVar*> mutate_vars;

  static SimpleOpr* CastFromBase(Opr* ptr);
};  // struct SimpleOpr

/*!
 * \brief Engine implementation.
 */
class SimpleEngine final : public DAGEngine {
 public:
  /*!
   * \brief Constructor and destructor.
   */
  SimpleEngine();
  ~SimpleEngine() noexcept(false);
  /*!
   * \brief Overriding methods.
   */
  Variable NewVar() override;
  Operator NewOperator(AsyncFn fn, std::vector<Variable> const& use_vars,
                       std::vector<Variable> const& mutate_vars) override;
  void DeleteOperator(Operator op) override;
  void Push(Operator op, Context exec_ctx) override;
  using DAGEngine::Push;
  void PushAsync(AsyncFn exec_fun, Context exec_ctx,
                 std::vector<Variable> const& use_vars,
                 std::vector<Variable> const& mutate_vars) override;
  void PushDelete(Fn delete_fn, Context exec_ctx, Variable var) override;
  void WaitForVar(Variable var) override;
  void WaitForAll() override;
  /*!
   * \brief Callback on operation completion.
   *
   * On operation completion, this will trigger subsequent operations.
   */
  void OnComplete(VersionedVarBlock* var);
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
   * \brief Number of pending operations.
   */
  std::atomic<std::size_t> pending_;
  std::condition_variable finished_cv_;
  std::mutex finished_m_;
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
  DISALLOW_COPY_AND_ASSIGN(SimpleEngine);
};  // class SimpleEngine

}  // namespace engine

}  // namespace mxnet

#endif  // MXNET_DAG_ENGINE_SIMPLE_ENGINE_H_
