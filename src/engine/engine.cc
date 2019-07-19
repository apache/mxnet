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

static inline std::string GetEngineType(bool* is_default = nullptr) {
  const char *type = getenv("MXNET_ENGINE_TYPE");
  if (is_default) {
    *is_default = (type == nullptr);
  }
  if (type == nullptr) type = "ThreadedEnginePerDevice";
  return std::string(type);
}

namespace engine {

inline Engine* CreateEngine() {
  bool default_engine = true;
  const std::string stype = GetEngineType(&default_engine);
  Engine *ret = nullptr;
  #if MXNET_PREDICT_ONLY == 0
  if (stype == "NaiveEngine" || stype == "NaiveEnginePerThread") {
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
    LOG(FATAL) << "Cannot find Engine " << stype;
  }
  if (!default_engine) {
    LOG(INFO) << "MXNet start using engine: " << stype;
  }
  return ret;
}
}  // namespace engine

static bool IsCreatePerThread() {
  std::string stype = GetEngineType();
  if (stype == "NaiveEnginePerThread") {
    return true;
  }
  return false;
}

std::shared_ptr<Engine> Engine::_GetSharedRef() {
  static bool per_thread = IsCreatePerThread();
  if (per_thread) {
#if DMLC_CXX11_THREAD_LOCAL
    static thread_local std::shared_ptr<Engine> sptr(engine::CreateEngine());
#else
    static MX_THREAD_LOCAL std::shared_ptr<Engine> sptr(engine::CreateEngine());
#endif
    return sptr;
  } else {
    static std::shared_ptr<Engine> sptr(engine::CreateEngine());
    return sptr;
  }
}



Engine *Engine::Get() {
  static bool per_thread = IsCreatePerThread();
  if (per_thread) {
#if DMLC_CXX11_THREAD_LOCAL
    static thread_local Engine *inst = _GetSharedRef().get();
#else
    static MX_THREAD_LOCAL Engine *inst = _GetSharedRef().get();
#endif
    return inst;
  } else {
    static Engine *inst = _GetSharedRef().get();
    return inst;
  }
}
}  // namespace mxnet
