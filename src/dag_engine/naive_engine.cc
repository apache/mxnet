// Copyright by Contributors
#include <mxnet/dag_engine.h>

namespace mxnet {
namespace engine {

// The Naive engine interface
class NaiveEngine : public DAGEngine {
 public:
  NaiveEngine() {
    #if MXNET_USE_CUDA
    #if MXNET_USE_CUDNN
    LOG(INFO) << "MXNET USE CUDNN";
    stream_ = mshadow::NewStream<gpu>(true, true);
    #else
    stream_ = mshadow::NewStream<gpu>(true, false);
    #endif // MXNET_USE_CUDNN
    ctx_.stream = stream_;
    #endif // MXNET_USE_CUDA
  }

  ~NaiveEngine() {
    #if MXNET_USE_CUDA
    mshadow::DeleteStream(stream_);
    #endif
  }

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
#if MXNET_USE_CUDA
      mshadow::SetDevice<gpu>(exec_ctx.dev_id);
      ctx_.stream = stream_;
      exec_fun(ctx_);
      stream_->Wait();
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

  void WaitToRead(Variable var) override {
  }

  void WaitToWrite(Variable var) override {
  }

  void WaitForAll() override {
  }

 private:
  RunContext ctx_;
  #if MXNET_USE_CUDA
  mshadow::Stream<gpu> *stream_;
  #endif
};

}  // namespace engine

DAGEngine* DAGEngine::Get() {
  static mxnet::engine::NaiveEngine engine;
  return &engine;
}
}  // namespace mxnet
