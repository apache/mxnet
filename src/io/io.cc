// Copyright (c) 2015 by Contributors

#include <mxnet/io.h>
#include <dmlc/registry.h>
#include "./image_augmenter.h"
#include "./image_iter_common.h"

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
DMLC_REGISTER_PARAMETER(ImageRecParserParam);
DMLC_REGISTER_PARAMETER(ImageRecordParam);
DMLC_REGISTER_PARAMETER(ImageDetNormalizeParam);
}  // namespace io
}  // namespace mxnet
