// Copyright (c) 2015 by Contributors
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <mxnet/io.h>
#include <dmlc/registry.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::DataIteratorReg);
}  // namespace dmlc
