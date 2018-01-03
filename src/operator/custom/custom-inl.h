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

  template<typename Func>
  void Push(const Func& func,
            const OpContext& ctx,
            bool recording,
            bool training,
            const std::vector<NDArray>& arrs) {
    if (naive_engine_) {
      func();
      ctx.async_on_complete();
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    q_.push(
      [=]() mutable {
        bool prev_recording = Imperative::Get()->set_is_recording(recording);
        bool prev_training = Imperative::Get()->set_is_training(training);

        func();

        Imperative::Get()->set_is_training(prev_training);
        Imperative::Get()->set_is_recording(prev_recording);

        std::vector<Engine::VarHandle> vars;
        for (const auto& i : arrs) vars.push_back(i.var());
        Engine::Get()->PushSync([=](RunContext rctx) {
            ctx.async_on_complete();
          }, ctx.run_ctx.ctx, vars, {},
          FnProperty::kNormal, 0, PROFILER_MESSAGE("CustomOperator"));
      });
    cv_.notify_all();
  }

  ~CustomOperator() {
    if (naive_engine_) return;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      destructing_ = true;
      cv_.notify_all();
    }
    worker_.join();
  }

  static CustomOperator* Get();

 private:
  CustomOperator() {
    destructing_ = false;
    naive_engine_ = true;
    if (std::string("NaiveEngine") != dmlc::GetEnv("MXNET_ENGINE_TYPE", std::string())) {
      naive_engine_ = false;
      worker_ = std::thread(
        [&]() {
          std::unique_lock<std::mutex> lock(mutex_);
          while (!q_.empty() || !destructing_) {
            cv_.wait(lock, [&] {return !q_.empty() || destructing_;});
            while (!q_.empty()) {
              auto fn = q_.front();
              lock.unlock();
              fn();
              lock.lock();
              q_.pop();
            }
          }
        });
    }
  }
  std::mutex mutex_;
  std::map<std::string, CustomOpPropCreator> registry_;
  // async worker
  std::condition_variable cv_;
  std::thread worker_;
  std::queue<std::function<void(void)> > q_;
  bool naive_engine_;
  bool destructing_;
};

}  // namespace custom
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CUSTOM_CUSTOM_INL_H_
