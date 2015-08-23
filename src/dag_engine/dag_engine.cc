/*!
 * Copyright (c) 2015 by Contributors
 */
#include "mxnet/dag_engine.h"
#include "simple_engine.h"

namespace mxnet {

void DAGEngine::Push(Fn exec_fun, Context exec_ctx,
                     std::vector<Variable> const& use_vars,
                     std::vector<Variable> const& mutate_vars) {
  auto f = [exec_fun](RunContext ctx, Callback on_complete) {
    exec_fun(ctx);
    on_complete();
  };
  PushAsync(f, exec_ctx, use_vars, mutate_vars);
}

DAGEngine::~DAGEngine() = default;

DAGEngine::DAGEngine() = default;

DAGEngine* DAGEngine::Get() {
  using EngineImplementation = engine::SimpleEngine;

  static EngineImplementation inst;
  return &inst;
}

}  // namespace mxnet
