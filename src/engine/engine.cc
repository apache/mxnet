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

  if (ret ==nullptr) {
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
}  // namespace mxnet
