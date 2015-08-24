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
#include "mxnet/dag_engine.h"
#include "dag_engine_impl.h"
#include "thread_pool.h"

namespace mxnet {

namespace engine {

struct SimpleOpr;
struct OprBlock;

struct VersionedVarBlock {
  VersionedVarBlock* next{nullptr};
  VersionedVarBlock* join{nullptr};
  OprBlock* trigger{nullptr};
  dmlc::Spinlock lock;
  bool waiting{false};
};  // struct VersionedVarBlock

struct OprBlock {
  std::function<void()> fn;
  std::atomic<std::size_t> wait{0};
};  // struct OprBlock

struct SimpleVar final : public Var {
  VersionedVarBlock* var{nullptr};

  static SimpleVar* CastFromBase(Var* ptr);
};  // struct SimpleVar

struct SimpleOpr final : public Opr {
  DAGEngine::AsyncFn fn;
  std::vector<SimpleVar*> use_vars;
  std::vector<SimpleVar*> mutate_vars;

  static SimpleOpr* CastFromBase(Opr* ptr);
};  // struct SimpleOpr

class SimpleEngine final : public DAGEngine {
 public:
  SimpleEngine();
  ~SimpleEngine() noexcept(false);
  Variable NewVar() override;
  Operator NewOperator(AsyncFn, std::vector<Variable> const&,
                       std::vector<Variable> const&) override;
  void DeleteOperator(Operator op) override;
  void Push(Operator op, Context) override;

  void PushAsync(AsyncFn, Context, std::vector<Variable> const&,
                 std::vector<Variable> const&) override{};

  void PushDelete(Fn, Context, Variable) override{};

  void WaitForVar(Variable) override{};

  void WaitForAll() override{};

  void OnComplete(VersionedVarBlock* var);
  void ThreadWorker();

 private:
  static constexpr std::size_t kNumWorkingThreads = 16;
  dmlc::ConcurrentBlockingQueue<OprBlock*> task_queue_;
  ThreadPool<kNumWorkingThreads> thread_pool_;
  DISALLOW_COPY_AND_ASSIGN(SimpleEngine);
};  // class SimpleEngine

}  // namespace engine

}  // namespace mxnet

#endif  // MXNET_DAG_ENGINE_SIMPLE_ENGINE_H_
