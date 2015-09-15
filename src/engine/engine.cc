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
  if (type == nullptr) type = "ThreadedEngine";
  std::string stype = type;
  Engine *ret = nullptr;
  if (stype == "ThreadedEngine") {
    ret = CreateThreadedEngine();
  } else if (stype == "NaiveEngine") {
    ret =  CreateNaiveEngine();
  }
  CHECK_NE(ret, nullptr)
      << "Cannot find Eine " << type << " in registry";
  if (!default_engine) {
    LOG(INFO) << "MXNet start using engine: " << type;
  }
  return ret;
}
}  // namespace engine

Engine* Engine::Get() {
  static std::unique_ptr<Engine> inst(engine::CreateEngine());
  return inst.get();
}
}  // namespace mxnet
