#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <mxnet/api_registry.h>

namespace mxnet {

FunctionRegistry::Entry &
FunctionRegistry::Register(const std::string name) {
  CHECK(fmap_.count(name) == 0);
  Entry *e = new Entry(name);
  fmap_[name] = e;
  fun_list_.push_back(e);
  // delete me later
  LOG(INFO) << "register function " << name;
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

}  // namespace mxnet
