/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_ENGINE_NAIVE_ENGINE_H_
#define MXNET_ENGINE_NAIVE_ENGINE_H_

#include <vector>
#include "engine_impl.h"

namespace mxnet {

namespace engine {

class NaiveEngine final : public Engine {
 public:
  NaiveEngine();
  ~NaiveEngine();
  VarHandle NewVariable() override;
  OprHandle NewOperator(AsyncFn fn, std::vector<VarHandle> const& const_vars,
                        std::vector<VarHandle> const& mutable_vars) override;
  void DeleteOperator(OprHandle op) override;
  void Push(OprHandle op, Context exec_ctx) override;
  void Push(Fn exec_fun, Context exec_ctx,
            std::vector<VarHandle> const& const_vars,
            std::vector<VarHandle> const& mutable_vars) override;
  void PushAsync(AsyncFn exec_fun, Context exec_ctx,
                 std::vector<VarHandle> const& const_vars,
                 std::vector<VarHandle> const& mutable_vars) override;
  void DeleteVariable(Fn delete_fun, Context exec_ctx, VarHandle var) override;
  void WaitForVar(VarHandle var) override;
  void WaitForAll() override;

 private:
  RunContext ctx_;
#if MXNET_USE_CUDA
  mshadow::Stream<gpu>* stream_;
#endif  // MXNET_USE_CUDA
};      // class NaiveEngine

}  // namespace engine

}  // namespace mxnet

#endif  // MXNET_ENGINE_NAIVE_ENGINE_H_
