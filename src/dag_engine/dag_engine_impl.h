/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_DAG_ENGINE_DAG_ENGINE_IMPL_H_
#define MXNET_DAG_ENGINE_DAG_ENGINE_IMPL_H_

#include <utility>
#include "mxnet/dag_engine.h"

#define DAG_ENGINE_DEBUG 0

namespace mxnet {
namespace engine {

struct Var {
#if DAG_ENGINE_DEBUG
  virtual ~Var() = default;
#endif  // DAG_ENGINE_DEBUG
  template <typename T>
  T* Cast();
};  // struct Var

struct Opr {
#if DAG_ENGINE_DEBUG
  virtual ~Opr() = default;
#endif  // DAG_ENGINE_DEBUG
  template <typename T>
  T* Cast();
};  // struct Opr

template <typename T>
T* Var::Cast() {
  static_assert(std::is_base_of<Var, T>::value,
                "must inherit `mxnet::engine::Var`");
#if DAG_ENGINE_DEBUG
  return dynamic_cast<T*>(this);
#else   // DAG_ENGINE_DEBUG
  return static_cast<T*>(this);
#endif  // DAG_ENGINE_DEBUG
}

template <typename T>
T* Opr::Cast() {
  static_assert(std::is_base_of<Opr, T>::value,
                "must inherit `mxnet::engine::Opr`");
#if DAG_ENGINE_DEBUG
  return dynamic_cast<T*>(this);
#else   // DAG_ENGINE_DEBUG
  return static_cast<T*>(this);
#endif  // DAG_ENGINE_DEBUG
}

}  // namespace engine
}  // namespace mxnet

#endif  // MXNET_DAG_ENGINE_DAG_ENGINE_IMPL_H_
