/*!
 * Copyright (c) 2015 by Contributors
 */
#include "./cpu_storage.h"
#include <dmlc/logging.h>
#include <cstdlib>

namespace mxnet {
namespace storage {

void* CpuStorage::Alloc(size_t size) {
  return CHECK_NOTNULL(aligned_alloc(alignment_, size));
}

void CpuStorage::Free(void* ptr) { free(ptr); }

}  // namespace storage
}  // namespace mxnet
