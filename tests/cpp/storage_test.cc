#include <cstdio>
#include <gtest/gtest.h>
#include <dmlc/logging.h>
#include <mxnet/storage.h>

extern bool unitTestsWithCuda;

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

static bool checkForWorkingCuda()
{
  int count = 0;
  if (cudaSuccess == cudaGetDeviceCount(&count)) {
    if (count == 0) return -1;
    for (int device = 0; device < count; ++device) {
      cudaDeviceProp prop;
      if (cudaSuccess == cudaGetDeviceProperties(&prop, device)) {
        std::printf("%d.%d ", prop.major, prop.minor);
        return true;
      }
    }
  }
  std::fprintf(stderr, "Warning: Could not find working CUDA driver\n");
  return false;
}

TEST(Storage, Basic_GPU) {
  if(unitTestsWithCuda || checkForWorkingCuda()) {
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
  }
}
#endif  // MXNET_USE_CUDA

