/*!
 *  Copyright (c) 2015 by Contributors
 * \file operator.cc
 * \brief operator module of mxnet
 */
#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <mxnet/operator.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::OperatorPropertyReg);
}  // namespace dmlc

namespace mxnet {
// implementation of all factory functions
OperatorProperty *OperatorProperty::Create(const char* type_name) {
  auto *creator = dmlc::Registry<OperatorPropertyReg>::Find(type_name);
  if (creator == nullptr) {
    LOG(FATAL) << "Cannot find Operator " << type_name << " in registry";
  }
  return creator->body();
}
}  // namespace mxnet
