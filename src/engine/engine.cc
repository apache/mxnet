/*!
 * Copyright (c) 2015 by Contributors
 * \file engine.cc
 * \brief Implementation of engine.
 */
#include <mxnet/engine.h>
#include <memory>
#include <cstdlib>
#include "./engine_impl.h"
#include "./profiler.h"

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
#if MXNET_USE_PROFILER
  static std::once_flag dump_profile_flag;
  std::call_once(dump_profile_flag, [](){
    // ensure profiler constructor is called before set atexit.
    engine::Profiler::Get();
    // dump trace file if profiler is enabled before engine is destructed.
    std::atexit([](){
      engine::Profiler* profiler = engine::Profiler::Get();
      if (profiler->IsEnableOutput()) {
        profiler->DumpProfile();
      }
    });
  });
#endif
  return inst;
}
}  // namespace mxnet
