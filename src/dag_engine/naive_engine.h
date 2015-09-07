/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_DAG_ENGINE_NAIVE_ENGINE_H_
#define MXNET_DAG_ENGINE_NAIVE_ENGINE_H_

#include <vector>
#include "dag_engine_impl.h"

namespace mxnet {

namespace engine {

class NaiveEngine final : public DAGEngine {
 public:
  NaiveEngine();
  ~NaiveEngine();
  Variable NewVar() override;
  OprHandle NewOperator(AsyncFn fn, std::vector<Variable> const& use_vars,
                        std::vector<Variable> const& mutate_vars) override;
  void DeleteOperator(OprHandle op) override;
  void Push(OprHandle op, Context exec_ctx) override;
  void Push(Fn exec_fun, Context exec_ctx,
            std::vector<Variable> const& use_vars,
            std::vector<Variable> const& mutate_vars) override;
  void PushAsync(AsyncFn exec_fun, Context exec_ctx,
                 std::vector<Variable> const& use_vars,
                 std::vector<Variable> const& mutate_vars) override;
  void PushDelete(Fn delete_fun, Context exec_ctx, Variable var) override;
  void WaitForVar(Variable var) override;
  void WaitForAll() override;

 private:
  RunContext ctx_;
#if MXNET_USE_CUDA
  mshadow::Stream<gpu>* stream_;
#endif
};  // class NaiveEngine

}  // namespace engine

}  // namespace mxnet

#endif  // MXNET_DAG_ENGINE_NAIVE_ENGINE_H_
