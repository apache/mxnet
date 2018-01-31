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
 * \file random_generator.cu
 * \brief gpu implements for parallel random number generator.
 */

#include <algorithm>
#include "./random_generator.h"
#include "../operator/mxnet_op.h"

namespace mxnet {
namespace common {
namespace random {

template<>
const int RandGenerator<gpu, float>::kMinNumRandomPerThread = 64;

template<>
const int RandGenerator<gpu, float>::kNumRandomStates = 32768;

__global__ void rand_generator_seed_kernel(curandStatePhilox4_32_10_t *states_,
                                           const int size,
                                           uint32_t seed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) curand_init(seed, id, 0, states_ + id);
}

template<>
void RandGenerator<gpu, float>::Seed(mshadow::Stream<gpu> *s, uint32_t seed) {
  using namespace mshadow::cuda;
  int ngrid = std::min(kMaxGridNum,
                       (RandGenerator<gpu, float>::kNumRandomStates + kBaseThreadNum - 1) /
                         kBaseThreadNum);
  rand_generator_seed_kernel
      <<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
          states_,
          RandGenerator<gpu, float>::kNumRandomStates,
          seed);
  MSHADOW_CUDA_POST_KERNEL_CHECK(rand_generator_seed_kernel);
  s->Wait();
}

}  // namespace random
}  // namespace common
}  // namespace mxnet
