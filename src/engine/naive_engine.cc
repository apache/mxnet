/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2015 by Contributors
 * \file naive_engine.cc
 * \brief Implementation of NaiveEngine
 */
#include <vector>
#include <atomic>
#include <thread>
#include "./engine_impl.h"
#include "./profiler.h"
#include "./openmp.h"

namespace mxnet {
namespace engine {

// implement naive engine
class NaiveEngine final : public Engine {
 public:
  struct NaiveOpr : public Opr {
    AsyncFn fn;
    std::vector<VarHandle> const_vars;
    std::vector<VarHandle> mutable_vars;
    FnProperty prop;
    const char* opr_name;
    /*! \brief indicate whether to profile this operator */
    bool profiling{false};
    /*! \brief operator execution statistics */
    OprExecStat *opr_stat;
  };

  NaiveEngine() {
  }
  // virtual destructor
  virtual ~NaiveEngine() {
#if MXNET_USE_CUDA
    LOG(INFO) << "Engine shutdown";
    for (size_t i = 0; i < streams_.size(); ++i) {
      if (streams_[i] != nullptr) {
        // Catch exception for CUDA driver shutdown
        MSHADOW_CATCH_ERROR(mshadow::DeleteStream(streams_[i]));
        streams_[i] = nullptr;
      }
    }
#endif
  }

  // new variables
  VarHandle NewVariable() override {
    size_t v = ++counter_;
    return reinterpret_cast<VarHandle>(v);
  }

  OprHandle NewOperator(AsyncFn fn,
                        std::vector<VarHandle> const& const_vars,
                        std::vector<VarHandle> const& mutable_vars,
                        FnProperty prop = FnProperty::kNormal,
                        const char* opr_name = nullptr) override {
    NaiveOpr *opr = new NaiveOpr();
    opr->fn = fn;
    opr->const_vars = const_vars;
    opr->mutable_vars = mutable_vars;
    opr->prop = prop;
    opr->opr_name = opr_name;
    return opr;
  }

  void DeleteOperator(OprHandle op) override {
    NaiveOpr *opr = op->Cast<NaiveOpr>();
    delete opr;
  }

  void Push(OprHandle op, Context exec_ctx, int priority = 0, bool profiling = false) override {
    Profiler *profiler = Profiler::Get();
    NaiveOpr *opr = op->Cast<NaiveOpr>();
    opr->profiling = profiling && (profiler->GetMode() == Profiler::kOnlySymbolic);
    this->PushAsync([&](RunContext ctx, CallbackOnComplete on_complete) {
#if MXNET_USE_PROFILER
        if (opr->profiling) {
          opr->opr_stat = Profiler::Get()->AddOprStat(exec_ctx.dev_type, exec_ctx.dev_id);
          uint64_t id = std::hash<std::thread::id>()(std::this_thread::get_id());
          opr->opr_stat->thread_id = id;
          strncpy(opr->opr_stat->opr_name,
            opr->opr_name,
            sizeof(opr->opr_stat->opr_name) - 1);
          SetOprStart(opr->opr_stat);
        }
        opr->fn(ctx, on_complete);
        if (opr->profiling) {
          SetOprEnd(opr->opr_stat);
        }
#else
        opr->fn(ctx, on_complete);
#endif
      },
      exec_ctx,
      opr->const_vars,
      opr->mutable_vars,
      opr->prop,
      priority,
      PROFILER_MESSAGE(opr->opr_name));
  }

  void PushAsync(AsyncFn exec_fun,
                 Context exec_ctx,
                 std::vector<VarHandle> const& const_vars,
                 std::vector<VarHandle> const& mutable_vars,
                 FnProperty prop = FnProperty::kNormal,
                 int priority = 0,
                 const char* opr_name = nullptr) override {
    CallbackOnComplete callback = CreateCallback(
        NaiveEngine::OnComplete, nullptr);
    this->req_completed_ = false;
#if MXNET_USE_PROFILER
    Profiler *profiler = Profiler::Get();
    NaiveOpr *opr = nullptr;
    bool profiling = (profiler->GetState() == Profiler::kRunning) &&
                   (profiler->GetMode() == Profiler::kAllOperator) &&
                   opr_name;
    if (profiling) {
      opr = NewOperator(exec_fun, const_vars, mutable_vars,
                        prop, opr_name)->Cast<NaiveOpr>();
      opr->profiling = profiling;
      opr->opr_stat = Profiler::Get()->AddOprStat(exec_ctx.dev_type, exec_ctx.dev_id);
      uint64_t id = std::hash<std::thread::id>()(std::this_thread::get_id());
      opr->opr_stat->thread_id = id;
      strncpy(opr->opr_stat->opr_name,
              opr->opr_name,
              sizeof(opr->opr_stat->opr_name) - 1);
      SetOprStart(opr->opr_stat);
    }
#endif
    if (exec_ctx.dev_mask() == gpu::kDevMask) {
#if MXNET_USE_CUDA
      size_t dev_id = static_cast<size_t>(exec_ctx.dev_id);
      MSHADOW_CATCH_ERROR(mshadow::SetDevice<gpu>(exec_ctx.dev_id));
      if (streams_.size() <= dev_id) {
        streams_.resize(dev_id + 1, nullptr);
      }
      if (streams_[dev_id] == nullptr) {
        streams_[dev_id] = mshadow::NewStream<gpu>(true, MXNET_USE_CUDNN != 0, dev_id);
      }
      exec_fun(RunContext{exec_ctx, streams_[dev_id]}, callback);
#else
      LOG(FATAL) << "GPU is not enabled";
#endif
    } else {
      exec_fun(RunContext{exec_ctx, &cpu_stream_}, callback);
    }
    CHECK(this->req_completed_)
        << "NaiveEngine only support synchronize Push so far";
#if MXNET_USE_PROFILER
    if (profiling) {
      SetOprEnd(opr->opr_stat);
    }
#endif
  }

  void DeleteVariable(SyncFn delete_fn, Context exec_ctx, VarHandle var) override {
    this->PushSync(delete_fn, exec_ctx, {}, {var},
                   FnProperty::kNormal, 0, PROFILER_MESSAGE("DeleteVariable"));
  }

  void WaitForVar(VarHandle var) override {
  }

  void WaitForAll() override {
  }

  void NotifyShutdown() override {
    shutdown_phase_.store(true);
  }

 private:
  // callback to oncomplete
  static void OnComplete(Engine *engine, void *param) {
    static_cast<NaiveEngine*>(engine)->req_completed_ = true;
  }
  // whether action is completed
  bool req_completed_;
  // counter
  std::atomic<size_t> counter_{0};
  /*! \brief whether it is during shutdown phase*/
  std::atomic<bool> shutdown_phase_{false};
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
