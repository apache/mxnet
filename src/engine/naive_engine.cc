/*!
 * Copyright (c) 2015 by Contributors
 */
#include "naive_engine.h"
#include <vector>

namespace mxnet {
namespace engine {

NaiveEngine::Variable NaiveEngine::NewVar() { return nullptr; }

NaiveEngine::NaiveEngine() {
#if MXNET_USE_CUDA
  stream_ = mshadow::NewStream<gpu>(true, false);
  ctx_.stream = stream_;
#endif
}

NaiveEngine::~NaiveEngine() {
#if MXNET_USE_CUDA
  mshadow::DeleteStream(stream_);
#endif
}

NaiveEngine::OprHandle NaiveEngine::NewOperator(AsyncFn,
                                                std::vector<Variable> const&,
                                                std::vector<Variable> const&) {
  LOG(FATAL) << "Not implemented";
  return nullptr;
}

void NaiveEngine::DeleteOperator(OprHandle) { LOG(FATAL) << "Not implemented"; }

void NaiveEngine::Push(OprHandle, Context) { LOG(FATAL) << "Not implemented"; }

void NaiveEngine::Push(Fn exec_fun, Context exec_ctx,
                       std::vector<Variable> const&,
                       std::vector<Variable> const&) {
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

void NaiveEngine::PushAsync(AsyncFn, Context, std::vector<Variable> const&,
                            std::vector<Variable> const&) {
  LOG(FATAL) << "Not implemented";
}

void NaiveEngine::PushDelete(Fn delete_fun, Context exec_ctx, Variable var) {
  this->Push(delete_fun, exec_ctx, {}, {var});
}

void NaiveEngine::WaitForVar(Variable) {}

void NaiveEngine::WaitForAll() {}

}  // namespace engine

}  // namespace mxnet
