/*!
 * Copyright (c) 2015 by Contributors
 */
#include "mxnet/dag_engine.h"
#include "simple_engine.h"
#include "dag_engine_impl.h"

namespace mxnet {

DAGEngine* DAGEngine::Get() {
  /*!
   * \brief Change specific engine to use.
   */
  using EngineImplementation = engine::SimpleEngine;

  static EngineImplementation inst;
  return &inst;
}

}  // namespace mxnet
