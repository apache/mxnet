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
 * Copyright (c) 2015 by Contributors
 * \file native_op-inl.h
 * \brief
 * \author Junyuan Xie
*/

#ifndef MXNET_OPERATOR_CUSTOM_CUSTOM_INL_H_
#define MXNET_OPERATOR_CUSTOM_CUSTOM_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/c_api.h>
#include <mxnet/imperative.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <sstream>
#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <queue>
#include "../operator_common.h"
#include "../../profiler/custom_op_profiler.h"

namespace mxnet {
namespace op {
namespace custom {

class CustomOperator {
 public:
  void Register(const std::string &op_type, CustomOpPropCreator creator) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (registry_.find(op_type) != registry_.end()) {
      LOG(WARNING) << "New registration is overriding existing custom operator " << op_type;
    }
    registry_[op_type] = creator;
  }

  CustomOpPropCreator Find(const std::string &op_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = registry_.find(op_type);
    if (it != registry_.end()) return it->second;
    return nullptr;
  }

  // For sparse the memory allocation is done during execution of operator
  // which leads to changing of the pointers stored by ndarray chunk.
  // Thus the changes to the copied ndarries don't propage to final
  // inputs and outputs unlike the dense case. Passing vector of inputs and
  // outputs ndarrays as args and updating the inputs and outputs ndarray
  // chunk pointers to be same as the copied ndarrays.
  template <typename Func>
  void Push(const Func& func, const OpContext& ctx, bool recording,
            bool training, const std::vector<NDArray>& arrs,
            const std::vector<int>& tags,
            const std::unordered_set<int>& output_tags,
            const std::vector<NDArray>& outputs,
            const std::string op_type = "") {
    if (naive_engine_) {
      if (profiler::Profiler::Get()->IsProfiling(profiler::Profiler::kImperative)) {
        profiler::CustomOpProfiler::Get()->OnCustomBegin(op_type);
        func();
        profiler::CustomOpProfiler::Get()->OnCustomEnd();
      } else {
        func();
      }
      for (size_t i = 0, out_idx = 0; i < arrs.size(); i++) {
        if (arrs[i].storage_type() == kDefaultStorage ||
            arrs[i].storage_type() == kUndefinedStorage)
          continue;
        if (output_tags.count(tags[i]) > 0) {
          outputs[out_idx].SparseUpdateChunk(arrs[i]);
          out_idx++;
        }
      }
      ctx.async_on_complete();
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    q_.push([=]() mutable {
      bool prev_recording = Imperative::Get()->set_is_recording(recording);
      bool prev_training = Imperative::Get()->set_is_training(training);

      try {
        if (profiler::Profiler::Get()->IsProfiling(profiler::Profiler::kImperative)) {
          profiler::CustomOpProfiler::Get()->OnCustomBegin(op_type);
          func();
          profiler::CustomOpProfiler::Get()->OnCustomEnd();
        } else {
          func();
        }
      } catch (dmlc::Error& e) {
        exception_ =
            std::make_shared<std::exception_ptr>(std::current_exception());
      }

      Imperative::Get()->set_is_training(prev_training);
      Imperative::Get()->set_is_recording(prev_recording);

      std::vector<Engine::VarHandle> vars, vars2;
      size_t idx = 0;
      for (const auto& i : arrs) {
        vars.push_back(i.var());
        if (output_tags.count(tags[idx]) > 0) {
          if (i.storage_type() == kDefaultStorage ||
              i.storage_type() == kUndefinedStorage)
            continue;
          vars2.push_back(i.var());
          idx++;
        }
      }

      Engine::Get()->PushSync(
          [=](RunContext rctx) {
            try {
              Throw();
              for (const auto& i : arrs) {
                Engine::Get()->Throw(i.var());
              }
            } catch(dmlc::Error& err) {
              ctx.async_on_complete(&err);
              return;
            }

            for (size_t i = 0, out_idx = 0; i < arrs.size(); i++) {
              if (arrs[i].storage_type() == kDefaultStorage ||
                  arrs[i].storage_type() == kUndefinedStorage)
                continue;
              if (output_tags.count(tags[i]) > 0) {
                outputs[out_idx].SparseUpdateChunk(arrs[i]);
                out_idx++;
              }
            }

            ctx.async_on_complete();
          },
          ctx.run_ctx.ctx, vars, vars2, FnProperty::kNoSkip, 0, "CustomOperatorWait");
    });
    // increase num_threads if there is not enough threads to execute custom operator
    if (q_.size() > num_free_threads_)
      CreateThreads(q_.size() - num_free_threads_);
    cv_.notify_all();
  }

  static CustomOperator* Get() {
    static CustomOperator inst;
    return &inst;
  }

  void Start() {
    num_free_threads_ = 0;
    destructing_ = false;
    naive_engine_ = true;
    exception_ = nullptr;
    if (std::string("NaiveEngine") != dmlc::GetEnv("MXNET_ENGINE_TYPE", std::string())) {
      naive_engine_ = false;
    }
  }

  void Stop() {
    if (naive_engine_) return;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      destructing_ = true;
      cv_.notify_all();
    }
    for (auto &worker : workers_)
      worker.join();
    workers_.clear();
  }

  inline void Throw() {
    if (exception_ && *exception_) {
      std::exception_ptr tmp = *exception_;
      exception_ = nullptr;
      std::rethrow_exception(tmp);
    }
  }

 private:
  CustomOperator() {
    this->Start();
  }
  void ThreadTarget() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!q_.empty() || !destructing_) {
      cv_.wait(lock, [&] {return !q_.empty() || destructing_;});
      while (!q_.empty()) {
        --num_free_threads_;
        auto fn = q_.front();
        q_.pop();
        lock.unlock();
        fn();
        ++num_free_threads_;
        lock.lock();
      }
    }
  }
  void SetNumThreads(int num_threads) {
    for (int i = workers_.size(); i < num_threads; ++i) {
      workers_.emplace_back(std::thread([this]{this->ThreadTarget();}));
      ++num_free_threads_;
    }
  }
  void CreateThreads(int num_new_threads) {
    SetNumThreads(workers_.size() + num_new_threads);
  }
  std::mutex mutex_;
  std::map<std::string, CustomOpPropCreator> registry_;
  // async worker
  std::condition_variable cv_;
  std::vector<std::thread> workers_;
  std::atomic<uint32_t> num_free_threads_;
  std::queue<std::function<void(void)> > q_;
  std::shared_ptr<std::exception_ptr> exception_;
  bool naive_engine_;
  bool destructing_;
};

}  // namespace custom
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CUSTOM_CUSTOM_INL_H_
