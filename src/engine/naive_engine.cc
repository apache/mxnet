/*!
 * Copyright (c) 2015 by Contributors
 */
#include "naive_engine.h"
#include <vector>

namespace mxnet {
namespace engine {

NaiveEngine::VarHandle NaiveEngine::NewVariable() { return nullptr; }

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
                                                std::vector<VarHandle> const&,
                                                std::vector<VarHandle> const&,
                                                FnProperty) {
  LOG(FATAL) << "Not implemented";
  return nullptr;
}

void NaiveEngine::DeleteOperator(OprHandle) { LOG(FATAL) << "Not implemented"; }

void NaiveEngine::Push(OprHandle, Context) { LOG(FATAL) << "Not implemented"; }

void NaiveEngine::Push(Fn exec_fun, Context exec_ctx,
                       std::vector<VarHandle> const&,
                       std::vector<VarHandle> const&, FnProperty) {
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

void NaiveEngine::PushAsync(AsyncFn, Context, std::vector<VarHandle> const&,
                            std::vector<VarHandle> const&, FnProperty) {
  LOG(FATAL) << "Not implemented";
}

void NaiveEngine::DeleteVariable(Fn delete_fun, Context exec_ctx,
                                 VarHandle var) {
  this->Push(delete_fun, exec_ctx, {}, {var}, FnProperty::kNormal);
}

void NaiveEngine::WaitForVar(VarHandle) {}

void NaiveEngine::WaitForAll() {}

}  // namespace engine

}  // namespace mxnet
