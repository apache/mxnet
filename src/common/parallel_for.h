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
 * Copyright (c) 2015 by Contributors
 */

#ifndef MXNET_COMMON_PARALLEL_FOR_H_
#define MXNET_COMMON_PARALLEL_FOR_H_

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

#include "../operator/mxnet_op.h"

namespace mxnet {
namespace common {

template <typename F>
void parallel_for(const size_t begin, const size_t end, const size_t grain_size, F&& f) {
  auto divup = [&](int64_t x, int64_t y) { return (x + y - 1) / y; };
  if (begin >= end) {
    return;
  }
#ifdef _OPENMP
  if (!omp_in_parallel() && ((end - begin) > grain_size) &&
      engine::OpenMP::Get()->GetRecommendedOMPThreadCount() > 1) {
#pragma omp parallel
    {
      int64_t num_threads = omp_get_num_threads();
      if (grain_size > 0) {
        num_threads = std::min(num_threads, divup((end - begin), grain_size));
      }
      auto tid = omp_get_thread_num();
      auto chunk_size = divup((end - begin), num_threads);
      auto begin_tid = begin + tid * chunk_size;
      if (begin_tid < end) {
        auto end_tid = std::min(end, chunk_size + begin_tid);
        f(begin_tid, end_tid);
      }
    }
  } else {
#endif
    f(begin, end);
  }
}

template <typename F>
void parallel_for(const size_t begin, const size_t end, F&& f) {
  constexpr int default_grain_size = 1;
  parallel_for(begin, end, default_grain_size, std::forward<F>(f));
}
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_COMMON_PARALLEL_FOR_H_
