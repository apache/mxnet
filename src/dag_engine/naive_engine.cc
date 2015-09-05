// Copyright by Contributors
#include <mxnet/dag_engine.h>

namespace mxnet {
namespace engine {

// The Naive engine interface
class NaiveEngine : public DAGEngine {
 public:
  Variable NewVar() override {
    return nullptr;
  }

  OprHandle NewOperator(AsyncFn fn,
                        std::vector<Variable> const& use_vars,
                        std::vector<Variable> const& mutate_vars) override {
    LOG(FATAL) << "Not implemented";
    return nullptr;
  }

  void DeleteOperator(OprHandle op) override {
    LOG(FATAL) << "Not implemented";
  }

  void Push(OprHandle op, Context exec_ctx) override {
    LOG(FATAL) << "Not implemented";
  }

  void Push(Fn exec_fun, Context exec_ctx,
            std::vector<Variable> const& use_vars,
            std::vector<Variable> const& mutate_vars) override {
    if (exec_ctx.dev_mask == gpu::kDevMask) {
      ctx_.stream = &stream_;
#if MXNET_USE_CUDA
      mshadow::SetDevice<gpu>(exec_ctx.dev_id);
      exec_fun(ctx_);
#else
      LOG(FATAL) << "GPU is not enabled";
#endif
    } else {
      exec_fun(ctx_);
    }
  }

  void PushAsync(AsyncFn exec_fun, Context exec_ctx,
                 std::vector<Variable> const& use_vars,
                 std::vector<Variable> const& mutate_vars) override {
    LOG(FATAL) << "Not implemented";
  }

  void PushDelete(Fn delete_fun, Context exec_ctx, Variable var) override {
    this->Push(delete_fun, exec_ctx, {}, {var});
  }

  void WaitForVar(Variable var) override {
  }

  void WaitForAll() override {
  }

 private:
  RunContext ctx_;
  mshadow::Stream<gpu> stream_;
};

}  // namespace engine

}  // namespace mxnet
