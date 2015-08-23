// Copyright (c) 2015 by Contributors
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <mxnet/io.h>
#include <dmlc/logging.h>
#include <dmlc/config.h>
#include <dmlc/registry.h>
#include <mshadow/tensor.h>
#include <string>
#include <vector>
#include <fstream>
#include "iter_mnist-inl.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::DataIteratorReg);
}  // namespace dmlc
