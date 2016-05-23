/*!
 *  Copyright (c) 2015 by Contributors
 * \file optimizer.cc
 * \brief optimizer module of mxnet
 * \author Junyuan Xie
 */
#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <mxnet/optimizer.h>
#include <mxnet/ndarray.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::OptimizerReg);
}  // namespace dmlc

namespace mxnet {
// implementation of all factory functions
Optimizer *Optimizer::Create(const char* type_name) {
  auto *creator = dmlc::Registry<OptimizerReg>::Find(type_name);
  if (creator == nullptr) {
    LOG(FATAL) << "Cannot find Optimizer " << type_name << " in registry";
  }
  return creator->body();
}
}  // namespace mxnet
