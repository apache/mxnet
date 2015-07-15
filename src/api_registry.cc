/*!
 *  Copyright (c) 2015 by Contributors
 * \file api_registry.cc
 * \brief
 */
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <mxnet/api_registry.h>

namespace mxnet {

FunctionRegistry::Entry &
FunctionRegistry::Register(const std::string name) {
  CHECK_EQ(fmap_.count(name), 0);
  Entry *e = new Entry(name);
  fmap_[name] = e;
  fun_list_.push_back(e);
  return *e;
}

FunctionRegistry::~FunctionRegistry() {
  for (auto p = fmap_.begin(); p != fmap_.end(); ++p) {
    delete p->second;
  }
}

FunctionRegistry *FunctionRegistry::Get() {
  static FunctionRegistry instance;
  return &instance;
}

// SymbolCreatorRegistry

SymbolCreatorRegistry::Entry&
SymbolCreatorRegistry::Register(const std::string& name) {
  CHECK_EQ(fmap_.count(name), 0);
  Entry *e = new Entry(name);
  fmap_[name] = e;
  fun_list_.push_back(e);
  return *e;
}

SymbolCreatorRegistry::~SymbolCreatorRegistry() {
  for (auto p = fmap_.begin(); p != fmap_.end(); ++p) {
    delete p->second;
  }
}

SymbolCreatorRegistry *SymbolCreatorRegistry::Get() {
  static SymbolCreatorRegistry instance;
  return &instance;
}

}  // namespace mxnet
