/*!
 * Copyright (c) 2015 by Contributors
 */
#include "mxnet/engine.h"
#include "engine_impl.h"
#include "naive_engine.h"
#include "threaded_engine.h"

namespace mxnet {

Engine::~Engine() noexcept(false) {}

Engine* Engine::Get() {
  /*!
   * \brief Change specific engine to use.
   */
#ifdef MXNET_USE_THREADED_ENGINE
  using EngineImplementation = engine::ThreadedEngine;
#else  // MXNET_USE_THREADED_ENGINE
#warning "Using naive engine.";
  using EngineImplementation = engine::NaiveEngine;
#endif  // MXNET_USE_THREADED_ENGINE

  static EngineImplementation inst;
  return &inst;
}

}  // namespace mxnet
