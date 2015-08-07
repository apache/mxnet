/*!
 *  Copyright (c) 2015 by Contributors
 * \file api_registry.cc
 * \brief
 */
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <mxnet/registry.h>
#include <mxnet/symbol.h>

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

#if DMLC_USE_CXX11
template NArrayFunctionEntry &Registry<NArrayFunctionEntry>::Register(const std::string& name);
template Registry<NArrayFunctionEntry> *Registry<NArrayFunctionEntry>::Get();
#endif

template AtomicSymbolEntry &Registry<AtomicSymbolEntry>::Register(const std::string& name);
template Registry<AtomicSymbolEntry> *Registry<AtomicSymbolEntry>::Get();

}  // namespace mxnet
