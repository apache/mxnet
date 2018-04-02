/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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

