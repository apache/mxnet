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
 * \file cuda_utils.cc
 * \brief Common CUDA utilities.
 */

#include <mxnet/base.h>
#include <mshadow/base.h>

#include <algorithm>

#include "cuda_utils.h"

#if MXNET_USE_CUDA

namespace mxnet {
namespace common {
namespace cuda {

namespace {
  bool IsPower2(size_t N) {
    return ((N & (N - 1)) == 0) && N != 0;
  }

  size_t RoundToPower2(size_t N) {
    size_t ret = 1;
    size_t copyN = N;
    while (N >= 2) {
      ret *= 2;
      N /= 2;
    }
    if (ret < copyN) {
      ret *= 2;
    }
    return ret;
  }
}  // namespace

int get_load_type(size_t N) {
  using namespace mshadow;
  if (N % 8 == 0) {
    return kFloat64;
  } else if (N % 4 == 0) {
    return kFloat32;
  } else if (N % 2 == 0) {
    return kFloat16;
  } else {
    return kUint8;
  }
}

int get_rows_per_block(size_t row_size, int num_threads_per_block) {
  const int warp_size = 32;
  CHECK(IsPower2(num_threads_per_block))
    << "Number of threads in a block must be power of 2 to use get_rows_per_block function";
  // How many read instructions should 1 thread at least do
  const int read_instructions = 2;
  const int desired_num_threads_per_row = (row_size + read_instructions - 1) / read_instructions;
  int desired_num_warps_per_row = (desired_num_threads_per_row + warp_size - 1) / warp_size;
  int actual_num_warps_per_row = std::min(desired_num_warps_per_row,
                                          num_threads_per_block / warp_size);
  // actual number of warps needs to be power of 2
  actual_num_warps_per_row = RoundToPower2(actual_num_warps_per_row);
  return num_threads_per_block / (warp_size * actual_num_warps_per_row);
}

}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA
