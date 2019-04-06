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
 * \file engine.cc
 * \brief Implementation of engine.
 */
#include <mxnet/engine.h>
#include <memory>
#include <cstdlib>
#include "./engine_impl.h"

namespace mxnet {
namespace engine {
inline Engine* CreateEngine() {
  const char *type = getenv("MXNET_ENGINE_TYPE");
  const bool default_engine = (type == nullptr);
  if (type == nullptr) type = "ThreadedEnginePerDevice";
  std::string stype = type;

  Engine *ret = nullptr;
  #if MXNET_PREDICT_ONLY == 0
  if (stype == "NaiveEngine") {
    ret = CreateNaiveEngine();
  } else if (stype == "ThreadedEngine") {
    ret = CreateThreadedEnginePooled();
  } else if (stype == "ThreadedEnginePerDevice") {
    ret = CreateThreadedEnginePerDevice();
  }
  #else
  ret = CreateNaiveEngine();
  #endif

  if (ret == nullptr) {
    LOG(FATAL) << "Cannot find Engine " << type;
  }
  if (!default_engine) {
    LOG(INFO) << "MXNet start using engine: " << type;
  }
  return ret;
}
}  // namespace engine

std::shared_ptr<Engine> Engine::_GetSharedRef() {
  static std::shared_ptr<Engine> sptr(engine::CreateEngine());
  return sptr;
}

Engine* Engine::Get() {
  static Engine *inst = _GetSharedRef().get();
  return inst;
}

void Engine::PushAsyncPtr(AsyncFnPtr exec_fn_ptr, void* param, FnPtrParamDeleter del,
                          Context exec_ctx, std::vector<VarHandle> const& const_vars,
                          std::vector<VarHandle> const& mutable_vars,
                          FnProperty prop, int priority,
                          const char* opr_name, bool wait) {
  AsyncFn exec_fn;
  if (del == nullptr) {
    exec_fn = [exec_fn_ptr, param](RunContext rctx,
                                   CallbackOnComplete on_complete) {
      exec_fn_ptr(rctx, on_complete, param);
    };
  } else {
    // Wrap param in a shared_ptr with del as deleter such that del will be
    // called when the lambda goes out of scope.
    std::shared_ptr<void> shared_param(param, del);
    exec_fn = [exec_fn_ptr, shared_param](RunContext rctx,
                                          CallbackOnComplete on_complete) {
      exec_fn_ptr(rctx, on_complete, shared_param.get());
    };
  }

  PushAsync(exec_fn, exec_ctx, const_vars, mutable_vars, prop, priority, opr_name, wait);
}

void Engine::PushSyncPtr(SyncFnPtr exec_fn_ptr, void* param, FnPtrParamDeleter del,
                         Context exec_ctx, std::vector<VarHandle> const& const_vars,
                         std::vector<VarHandle> const& mutable_vars,
                         FnProperty prop, int priority, const char* opr_name) {
  SyncFn exec_fn;
  if (del == nullptr) {
    exec_fn = [exec_fn_ptr, param](RunContext rctx) {
      exec_fn_ptr(rctx, param);
    };
  } else {
    // Wrap param in a shared_ptr with del as deleter such that del will be
    // called when the lambda goes out of scope.
    std::shared_ptr<void> shared_param(param, del);
    exec_fn = [exec_fn_ptr, shared_param](RunContext rctx) {
      exec_fn_ptr(rctx, shared_param.get());
    };
  }

  PushSync(exec_fn, exec_ctx, const_vars, mutable_vars, prop, priority, opr_name);
}
}  // namespace mxnet
