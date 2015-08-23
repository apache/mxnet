/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_DAG_ENGINE_DAG_ENGINE_IMPL_H_
#define MXNET_DAG_ENGINE_DAG_ENGINE_IMPL_H_

#include <vector>
#include "mxnet/dag_engine.h"

namespace mxnet {
namespace engine {

struct Var {};  // struct Var

struct Opr {
  DAGEngine::AsyncFn fn;
  std::vector<Var*> use_vars;
  std::vector<Var*> mutate_vars;
};  // struct Opr

}  // namespace engine
}  // namespace mxnet

#endif  // MXNET_DAG_ENGINE_DAG_ENGINE_IMPL_H_
