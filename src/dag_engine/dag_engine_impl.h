/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_DAG_ENGINE_DAG_ENGINE_IMPL_H_
#define MXNET_DAG_ENGINE_DAG_ENGINE_IMPL_H_

#include <utility>
#include "mxnet/dag_engine.h"

namespace mxnet {
namespace engine {

struct Var {
  virtual ~Var() = default;

  template <typename T>
  T* Cast();
};  // struct Var

struct Opr {
  virtual ~Opr() = default;
  template <typename T>
  T* Cast();
};  // struct Opr

template <typename T>
T* Var::Cast() {
  static_assert(std::is_base_of<Var, T>::value, "must inherit `mxnet::engine::Var`");
#ifdef NDEBUG
  return reinterpret_cast<T*>(this);
#else  // NDEBUG
  return dynamic_cast<T*>(this);
#endif  // NDEBUG
}

template <typename T>
T* Opr::Cast() {
  static_assert(std::is_base_of<Opr, T>::value, "must inherit `mxnet::engine::Opr`");
#ifdef NDEBUG
  return reinterpret_cast<T*>(this);
#else  // NDEBUG
  return dynamic_cast<T*>(this);
#endif  // NDEBUG
}

}  // namespace engine
}  // namespace mxnet

#endif  // MXNET_DAG_ENGINE_DAG_ENGINE_IMPL_H_
