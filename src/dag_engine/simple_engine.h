/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_DAG_ENGINE_SIMPLE_ENGINE_H_
#define MXNET_DAG_ENGINE_SIMPLE_ENGINE_H_

#include <vector>
#include "mxnet/dag_engine.h"

namespace mxnet {

namespace engine {

class Variable {};

class Operator {};

class SimpleEngine final : public DAGEngine {
 public:
  Variable NewVar() override { return new engine::Variable; }

  Operator NewOperator(AsyncFn, std::vector<Variable> const&,
                       std::vector<Variable> const&) override {
    return new engine::Operator;
  }

  void DeleteOperator(Operator op) override {
    delete op;
  }

  void Push(Operator op, Context) override{};

  void PushAsync(AsyncFn, Context, std::vector<Variable> const&,
                 std::vector<Variable> const&) override{};

  void PushDelete(Fn, Context, Variable) override {};

  void WaitForVar(Variable) override {};

  void WaitForAll() override {};
};

}  // namespace engine

}  // namespace mxnet

#endif  // MXNET_DAG_ENGINE_SIMPLE_ENGINE_H_
