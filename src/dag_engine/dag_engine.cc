/*!
 * Copyright (c) 2015 by Contributors
 */
#include "mxnet/dag_engine.h"
#include "dag_engine_impl.h"
#include "naive_engine.h"
#include "threaded_engine.h"

namespace mxnet {

DAGEngine* DAGEngine::Get() {
  /*!
   * \brief Change specific engine to use.
   */
  using EngineImplementation = engine::NaiveEngine;

  static EngineImplementation inst;
  return &inst;
}

}  // namespace mxnet
