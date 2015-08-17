/*!
 * Copyright (c) 2015 by Contributors
 * \file test_storage.cc
 * \brief Test for storage.
 */
#include <cstdio>
#include <cassert>
#include "mxnet/storage.h"

int main() {
  constexpr size_t kSize = 1024;
  auto&& storage = mxnet::Storage::Get();
  mxnet::Context context_cpu{};
  auto&& handle = storage->Alloc(kSize, context_cpu);
  assert(handle.ctx == context_cpu);
  assert(handle.size == kSize);
  auto ptr = handle.dptr;
  storage->Free(handle);
  handle = storage->Alloc(kSize, context_cpu);
  assert(handle.ctx == context_cpu);
  assert(handle.size == kSize);
  assert(handle.dptr == ptr);
  printf("Success on CPU!\n");

#if MXNET_USE_CUDA
  mxnet::Context context_gpu{mxnet::gpu::kDevMask, 0};
  handle = storage->Alloc(kSize, context_gpu);
  assert(handle.ctx == context_gpu);
  assert(handle.size == kSize);
  ptr = handle.dptr;
  storage->Free(handle);
  handle = storage->Alloc(kSize, context_gpu);
  assert(handle.ctx == context_gpu);
  assert(handle.size == kSize);
  assert(handle.dptr == ptr);
  printf("Success on GPU!\n");
#endif  // MXNET_USE_CUDA
  return 0;
}
