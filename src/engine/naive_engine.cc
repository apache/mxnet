/*!
 *  Copyright (c) 2015 by Contributors
 * \file naive_engine.cc
 * \brief Implementation of NaiveEngine
 */
#include <vector>
#include "./engine_impl.h"

namespace mxnet {
namespace engine {
// implement naive engine
class NaiveEngine final : public Engine {
 public:
  NaiveEngine() {
  }
  // virtual destructor
  virtual ~NaiveEngine() {
#if MXNET_USE_CUDA
    LOG(INFO) << "Engine shutdown";
    for (size_t i = 0; i < streams_.size(); ++i) {
      if (streams_[i] != nullptr) {
        // Catch exception for CUDA driver shutdown
        try {
          mshadow::DeleteStream(streams_[i]);
        } catch (const dmlc::Error &e) {
          std::string what = e.what();
          if (what.find("driver shutting down") == std::string::npos) {
            LOG(ERROR) << "Ignore Error " << what << " during worker finalization";
          }
        }
        streams_[i] = nullptr;
      }
    }
#endif
  }

  // new variables
  VarHandle NewVariable() override {
    return nullptr;
  }
  OprHandle NewOperator(AsyncFn fn,
                     std::vector<VarHandle> const& const_vars,
                     std::vector<VarHandle> const& mutable_vars,
                     FnProperty prop) override {
    LOG(FATAL) << "Not implemented";
    return nullptr;
  }
  void DeleteOperator(OprHandle op) override {
    LOG(FATAL) << "Not implemented";
  }
  void Push(OprHandle op, Context exec_ctx) override {
    LOG(FATAL) << "Not implemented";
  }
  void PushAsync(AsyncFn exec_fun,
                 Context exec_ctx,
                 std::vector<VarHandle> const& const_vars,
                 std::vector<VarHandle> const& mutable_vars,
                 FnProperty prop) override {
    CallbackOnComplete callback = CreateCallback(
        NaiveEngine::OnComplete, nullptr);
    this->req_completed_ = false;

    if (exec_ctx.dev_mask() == gpu::kDevMask) {
#if MXNET_USE_CUDA
      size_t dev_id = static_cast<size_t>(exec_ctx.dev_id);
      try {
        mshadow::SetDevice<gpu>(exec_ctx.dev_id);
      } catch (const dmlc::Error &e) {
        std::string what = e.what();
        if (what.find("driver shutting down") == std::string::npos) {
          LOG(ERROR) << "Ignore Error " << what << " during worker finalization";
        }
      }
      if (streams_.size() <= dev_id) {
        streams_.resize(dev_id + 1, nullptr);
      }
      if (streams_[dev_id] == nullptr) {
        streams_[dev_id] = mshadow::NewStream<gpu>(true, MXNET_USE_CUDNN != 0);
      }
      ctx_.stream = streams_[dev_id];
      exec_fun(ctx_, callback);
#else
      LOG(FATAL) << "GPU is not enabled";
#endif
    } else {
      ctx_.stream = &cpu_stream_;
      exec_fun(ctx_, callback);
    }
    CHECK(this->req_completed_)
        << "NaiveEngine only support synchronize Push so far";
  }
  void DeleteVariable(SyncFn delete_fn, Context exec_ctx, VarHandle var) override {
    this->PushSync(delete_fn, exec_ctx, {}, {var}, FnProperty::kNormal);
  }
  void WaitForVar(VarHandle var) override {
  }
  void WaitForAll() override {
  }

 private:
  // callback to oncomplete
  static void OnComplete(Engine *engine, void *param) {
    static_cast<NaiveEngine*>(engine)->req_completed_ = true;
  }
  // runtime contetxt
  RunContext ctx_;
  // whether action is completed
  bool req_completed_;
  // CPU stream
  mshadow::Stream<cpu> cpu_stream_;
  // GPU streams
  std::vector<mshadow::Stream<gpu>*> streams_;
};  // class NaiveEngine


Engine *CreateNaiveEngine() {
  return new NaiveEngine();
}
}  // namespace engine
}  // namespace mxnet
