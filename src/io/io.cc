// Copyright (c) 2015 by Contributors

#include <mxnet/io.h>
#include <dmlc/registry.h>
#include "./image_augmenter.h"
#include "./iter_normalize.h"
#include "./iter_batchloader.h"
#include "./iter_prefetcher.h"

// Registers
namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::DataIteratorReg);
}  // namespace dmlc

namespace mxnet {
namespace io {
// Register parameters in header files
DMLC_REGISTER_PARAMETER(BatchParam);
DMLC_REGISTER_PARAMETER(PrefetcherParam);
DMLC_REGISTER_PARAMETER(ImageNormalizeParam);
}  // namespace io
}  // namespace mxnet
