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
  using EngineImplementation = engine::ThreadedEngine;

  static EngineImplementation inst;
  return &inst;
}

}  // namespace mxnet
