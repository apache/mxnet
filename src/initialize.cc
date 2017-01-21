/*!
 *  Copyright (c) 2016 by Contributors
 * \file initialize.cc
 * \brief initialize mxnet library
 */
#include <dmlc/logging.h>

namespace mxnet {

class LibraryInitializer {
 public:
  LibraryInitializer() {
    dmlc::InitLogging("mxnet");
  }
};

static LibraryInitializer __library_init;
}  // namespace mxnet

