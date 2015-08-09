/*!
 * Copyright (c) 2015 by Contributors
 */
#include "./gpu_storage.h"
#include "mxnet/cuda_utils.h"
#ifdef MXNET_USE_CUDA
#include <cuda_runtime.h>
#endif  // MXNET_USE_CUDA

namespace mxnet {
namespace storage {

void* GpuStorage::Alloc(size_t size) {
  void* ret;
#ifdef MXNET_USE_CUDA
  CUDA_CALL(cudaMalloc(&ret, size));
#else   // MXNET_USE_CUDA
  LOG(FATAL) << "Please compile with CUDA enabled";
#endif  // MXNET_USE_CUDA
  return ret;
}

void GpuStorage::Free(void* ptr) {
#ifdef MXNET_USE_CUDA
  CUDA_CALL(cudaFree(ptr));
#else   // MXNET_USE_CUDA
  LOG(FATAL) << "Please compile with CUDA enabled";
#endif  // MXNET_USE_CUDA
}

}  // namespace storage
}  // namespace mxnet
