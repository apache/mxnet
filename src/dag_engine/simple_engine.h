/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_DAG_ENGINE_SIMPLE_ENGINE_H_
#define MXNET_DAG_ENGINE_SIMPLE_ENGINE_H_

#include <vector>
#include <functional>
#include "mxnet/dag_engine.h"
#include "dag_engine_impl.h"

namespace mxnet {

namespace engine {

struct OprBlock;

struct VersionedVarBlock {
  VersionedVarBlock* next = nullptr;
  OprBlock* waiting = nullptr;
};  // struct VersionedVarBlock

struct OprBlock {
  std::function<void()> fn;
  VersionedVarBlock* trigger;
  Opr* opr;
};  // struct OprBlock

class SimpleEngine final : public DAGEngine {
 public:
  SimpleEngine();
  ~SimpleEngine();
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

 private:
  DISALLOW_COPY_AND_ASSIGN(SimpleEngine);
};  // class SimpleEngine

}  // namespace engine

}  // namespace mxnet

#endif  // MXNET_DAG_ENGINE_SIMPLE_ENGINE_H_
