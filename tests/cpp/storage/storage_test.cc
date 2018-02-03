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
#include <mxnet/storage.h>
#include "test_util.h"

namespace {
constexpr std::size_t size = 1024;
}

TEST(Storage, Basic_CPU) {

  auto storage = mxnet::Storage::Get();
  mxnet::Context context_cpu {};

  auto handle = storage->Alloc(size, mxnet::Context());
  EXPECT_EQ(handle->ctx, context_cpu);
  EXPECT_EQ(handle->size, size);
  handle.reset();

  handle = storage->Alloc(size, mxnet::Context());
  EXPECT_EQ(handle->ctx, context_cpu);
  EXPECT_EQ(handle->size, size);
  handle.reset();
}

#if MXNET_USE_CUDA

TEST(Storage, Basic_GPU) {
  if (!mxnet::test::unitTestsWithCuda) {
    return;
  }

  auto storage = mxnet::Storage::Get();
  mxnet::Context context_gpu = mxnet::Context::GPU(0);

  auto handle = storage->Alloc(size, mxnet::Context());
  EXPECT_EQ(handle->ctx, context_gpu);
  EXPECT_EQ(handle->size, size);
  handle.reset();

  handle = storage->Alloc(size, mxnet::Context());
  EXPECT_EQ(handle->ctx, context_gpu);
  EXPECT_EQ(handle->size, size);
  handle.reset();
}

#endif  // MXNET_USE_CUDA

