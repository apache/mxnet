/*!
 *  Copyright (c) 2016 by Contributors
 * \file initialize.cc
 * \brief initialize mxnet library
 */
#include <dmlc/logging.h>
#include <mxnet/engine.h>

#include "engine/profiler.h"

namespace mxnet {

class LibraryInitializer {
 public:
  LibraryInitializer() {
    dmlc::InitLogging("mxnet");
#if MXNET_USE_PROFILER
    // ensure profiler's constructor are called before atexit.
    engine::Profiler::Get();
    // DumpProfile will be called before engine's and profiler's destructor.
    std::atexit([](){
      engine::Profiler* profiler = engine::Profiler::Get();
      if (profiler->IsEnableOutput()) {
        profiler->DumpProfile();
      }
    });
#endif
  }
};

static LibraryInitializer __library_init;
}  // namespace mxnet

