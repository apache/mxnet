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
 * \file engine.cc
 * \brief Implementation of engine.
 */
#include <mxnet/engine.h>
#include <memory>
#include <cstdlib>
#include "./engine_impl.h"
#include "../common/cuda/utils.h"

namespace mxnet {
namespace engine {
inline Engine* CreateEngine() {
  const char* type          = getenv("MXNET_ENGINE_TYPE");
  const bool default_engine = (type == nullptr);
  if (type == nullptr)
    type = "ThreadedEnginePerDevice";
  std::string stype = type;

  // The async tag is used later to determine if we use the GPU dependecy engine
  std::string async_engine_tag = "Async";
  auto tag_pos                 = stype.find(async_engine_tag);
  if (tag_pos != std::string::npos && tag_pos + async_engine_tag.length() == stype.length()) {
    stype = stype.substr(0, tag_pos);
  }

  Engine* ret = nullptr;
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

#if MXNET_USE_CUDA
CUDAEvent::CUDAEvent(Context const& ctx)
    : event_(std::make_shared<cudaEvent_t>()), dev_id_(ctx.dev_id) {
  cudaEvent_t ev;
  common::cuda::DeviceStore device_store(dev_id_);
  CUDA_CALL(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  *event_ = ev;
}

CUDAEvent::~CUDAEvent() {
  if (event_ && *event_ != nullptr) {
    common::cuda::DeviceStore device_store(dev_id_);
    CUDA_CALL(cudaEventSynchronize(*event_));
    CUDA_CALL(cudaEventDestroy(*event_));
  }
}
#endif
}  // namespace engine

const std::shared_ptr<Engine>& Engine::_GetSharedRef() {
  static std::shared_ptr<Engine> sptr(engine::CreateEngine());
  return sptr;
}

Engine* Engine::Get() {
  static Engine* inst = _GetSharedRef().get();
  return inst;
}
}  // namespace mxnet
