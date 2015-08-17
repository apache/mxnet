/*!
 *  Copyright (c) 2015 by Contributors
 * \file api_registry.cc
 * \brief
 */
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <mxnet/registry.h>
#include <mxnet/symbolic.h>

namespace mxnet {

template <typename Entry>
Entry &Registry<Entry>::Register(const std::string& name) {
  CHECK_EQ(fmap_.count(name), 0);
  Entry *e = new Entry(name);
  fmap_[name] = e;
  fun_list_.push_back(e);
  return *e;
}

template <typename Entry>
Registry<Entry> *Registry<Entry>::Get() {
  static Registry<Entry> instance;
  return &instance;
}


template NArrayFunctionEntry &Registry<NArrayFunctionEntry>::Register(const std::string& name);
template Registry<NArrayFunctionEntry> *Registry<NArrayFunctionEntry>::Get();

template OperatorPropertyEntry &Registry<OperatorPropertyEntry>::Register(const std::string& name);
template Registry<OperatorPropertyEntry> *Registry<OperatorPropertyEntry>::Get();

// implementation of all factory functions
OperatorProperty *OperatorProperty::Create(const char* type_name) {
  auto *creator = Registry<OperatorPropertyEntry>::Find(type_name);
  CHECK_NE(creator, nullptr)
      << "Cannot find Operator " << type_name << " in registry";
  return (*creator)();
}
}  // namespace mxnet
