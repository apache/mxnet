/*!
 * Copyright (c) 2015 by Contributors
 */
#define _ISOC11_SOURCE
#include "./cpu_storage.h"
#include <dmlc/logging.h>
#include <stdlib.h>

namespace mxnet {
namespace storage {

void* CpuStorage::Alloc(size_t size) {
  return CHECK_NOTNULL(aligned_alloc(alignment_, size));
}

void CpuStorage::Free(void* ptr) { free(ptr); }

}  // namespace storage
}  // namespace mxnet
