/*!
 * Copyright (c) 2017 by Contributors
 * \file storage_test.cc
 * \brief cpu/gpu storage tests
*/
#include <gtest/gtest.h>
#include <dmlc/logging.h>
#include <mxnet/storage.h>
#include <cstdio>
#include "test_util.h"

TEST(Storage, Basic_CPU) {
  constexpr size_t kSize = 1024;
  auto&& storage = mxnet::Storage::Get();
  mxnet::Context context_cpu{};
  auto&& handle = storage->Alloc(kSize, context_cpu);
  EXPECT_EQ(handle.ctx, context_cpu);
  EXPECT_EQ(handle.size, kSize);
  storage->Free(handle);
  handle = storage->Alloc(kSize, context_cpu);
  EXPECT_EQ(handle.ctx, context_cpu);
  EXPECT_EQ(handle.size, kSize);
  storage->Free(handle);
}

#if MXNET_USE_CUDA
TEST(Storage, Basic_GPU) {
  if (mxnet::test::unitTestsWithCuda) {
    constexpr size_t kSize = 1024;
    mxnet::Context context_gpu = mxnet::Context::GPU(0);
    auto &&storage = mxnet::Storage::Get();
    auto &&handle = storage->Alloc(kSize, context_gpu);
    assert(handle.ctx == context_gpu);
    assert(handle.size == kSize);
    auto ptr = handle.dptr;
    storage->Free(handle);
    handle = storage->Alloc(kSize, context_gpu);
    EXPECT_EQ(handle.ctx, context_gpu);
    EXPECT_EQ(handle.size, kSize);
    EXPECT_EQ(handle.dptr, ptr);
    storage->Free(handle);
  }
}
#endif  // MXNET_USE_CUDA

