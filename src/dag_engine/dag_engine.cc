/*!
 * Copyright (c) 2015 by Contributors
 */
#include "mxnet/dag_engine.h"
#include "dag_engine_impl.h"
#include "threaded_engine.h"

namespace mxnet {

DAGEngine* DAGEngine::Get() {
  /*!
   * \brief Change specific engine to use.
   */
  using EngineImplementation = engine::ThreadedEngine;

  static EngineImplementation inst;
  return &inst;
}

}  // namespace mxnet
