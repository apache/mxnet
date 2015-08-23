/*!
 * Copyright (c) 2015 by Contributors
 */
#include "simple_engine.h"
#include <mshadow/tensor.h>
#include <dmlc/logging.h>

namespace mxnet {

namespace engine {

SimpleEngine::SimpleEngine() = default;

SimpleEngine::~SimpleEngine() = default;

SimpleEngine::Variable SimpleEngine::NewVar() { return new Var{}; }

SimpleEngine::Operator SimpleEngine::NewOperator(
    SimpleEngine::AsyncFn fn, std::vector<Variable> const& use_vars,
    std::vector<Variable> const& mutate_vars) {
  return new Opr{fn, use_vars, mutate_vars};
}

void SimpleEngine::DeleteOperator(Operator op) {
  delete op;
}

void SimpleEngine::Push(Operator op, Context exec_ctx) {
}

}  // namespace engine

}  // namespace mxnet
