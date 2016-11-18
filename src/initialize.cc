/*!
 *  Copyright (c) 2016 by Contributors
 * \file initialize.cc
 * \brief initialize mxnet library
 */
#include <dmlc/logging.h>

namespace mxnet {
namespace op {
void RegisterLegacyOpProp();
void RegisterLegacyNDFunc();
}

class LibraryInitializer {
 public:
  LibraryInitializer() {
    dmlc::InitLogging("mxnet");
    mxnet::op::RegisterLegacyOpProp();
    mxnet::op::RegisterLegacyNDFunc();
  }
};

static LibraryInitializer __library_init;
}  // namespace mxnet

