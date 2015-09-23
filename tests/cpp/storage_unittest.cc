#include <cstdio>
#include <gtest/gtest.h>
#include <dmlc/logging.h>
#include <mxnet/storage.h>

TEST(Storage, basics) {
  constexpr size_t kSize = 1024;
  auto&& storage = mxnet::Storage::Get();
  mxnet::Context context_cpu{};
  auto&& handle = storage->Alloc(kSize, context_cpu);
  EXPECT_EQ(handle.ctx, context_cpu);
  EXPECT_EQ(handle.size, kSize);
  auto ptr = handle.dptr;
  storage->Free(handle);
  handle = storage->Alloc(kSize, context_cpu);
  EXPECT_EQ(handle.ctx, context_cpu);
  EXPECT_EQ(handle.size, kSize);
  EXPECT_EQ(handle.dptr, ptr);
  LOG(INFO) << "Success on CPU!\n";

#if MXNET_USE_CUDA
  mxnet::Context context_gpu = mxnet::Context::GPU(0);
  handle = storage->Alloc(kSize, context_gpu);
  assert(handle.ctx == context_gpu);
  assert(handle.size == kSize);
  ptr = handle.dptr;
  storage->Free(handle);
  handle = storage->Alloc(kSize, context_gpu);
  EXPECT_EQ(handle.ctx, context_gpu);
  EXPECT_EQ(handle.size, kSize);
  EXPECT_EQ(handle.dptr, ptr);
  LOG(INFO) << "Success on GPU!\n";
#endif  // MXNET_USE_CUDA
}
