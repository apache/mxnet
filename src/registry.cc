/*!
 *  Copyright (c) 2015 by Contributors
 * \file registry.cc
 * \brief central place for registry definition in mxnet.
 */
#include <dmlc/base.h>
#include <dmlc/registry.h>
#include <mxnet/registry.h>
#include <mxnet/symbolic.h>

// enable the registries
namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::NArrayFunctionReg);
DMLC_REGISTRY_ENABLE(::mxnet::OperatorPropertyReg);
}  // namespace dmlc

namespace mxnet {
// implementation of all factory functions
OperatorProperty *OperatorProperty::Create(const char* type_name) {
  auto *creator = dmlc::Registry<OperatorPropertyReg>::Find(type_name);
  CHECK_NE(creator, nullptr)
      << "Cannot find Operator " << type_name << " in registry";
  return creator->body();
}
}  // namespace mxnet
