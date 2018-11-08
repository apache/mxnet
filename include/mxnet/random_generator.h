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
 * \file random_generator.h
 * \brief Parallel random number generator.
 */
#ifndef MXNET_RANDOM_GENERATOR_H_
#define MXNET_RANDOM_GENERATOR_H_

#include <random>
#include <new>
#include "./base.h"

#if MXNET_USE_CUDA
#include <curand_kernel.h>
#endif  // MXNET_USE_CUDA

namespace mxnet {
namespace common {
namespace random {

template<typename Device, typename DType MSHADOW_DEFAULT_DTYPE>
class RandGenerator;

template<typename DType>
class RandGenerator<cpu, DType> {
 public:
  // at least how many random numbers should be generated by one CPU thread.
  static const int kMinNumRandomPerThread;
  // store how many global random states for CPU.
  static const int kNumRandomStates;

  // implementation class for random number generator
  // TODO(alexzai): move impl class to separate file - tracked in MXNET-948
  class Impl {
   public:
    typedef typename std::conditional<std::is_floating_point<DType>::value,
                                      DType, double>::type FType;
    typedef typename std::conditional<std::is_integral<DType>::value,
                                      DType, int>::type IType;
    explicit Impl(RandGenerator<cpu, DType> *gen, int state_idx)
        : engine_(gen->states_ + state_idx) {}

    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;

    MSHADOW_XINLINE int rand() { return engine_->operator()(); }

    MSHADOW_XINLINE FType uniform() {
      typedef typename std::conditional<std::is_integral<DType>::value,
      std::uniform_int_distribution<DType>,
      std::uniform_real_distribution<FType>>::type GType;
      GType dist_uniform;
      return dist_uniform(*engine_);
    }

    MSHADOW_XINLINE IType discrete_uniform(const int64_t lower, const int64_t upper) {
      typedef typename std::conditional<sizeof(IType) != sizeof(int32_t) ||
      sizeof(IType) != sizeof(int64_t),
      std::uniform_int_distribution<int>,
      std::uniform_int_distribution<IType>>::type GType;
      GType dist_discrete_uniform(lower, upper);
      return dist_discrete_uniform(*engine_);
    }

    MSHADOW_XINLINE FType normal() {
      std::normal_distribution<FType> dist_normal;
      return dist_normal(*engine_);
    }

   private:
    std::mt19937 *engine_;
  };  // class RandGenerator<cpu, DType>::Impl

  static void AllocState(RandGenerator<cpu, DType> *inst) {
    inst->states_ = new std::mt19937[kNumRandomStates];
  }

  static void FreeState(RandGenerator<cpu, DType> *inst) {
    delete[] inst->states_;
  }

  MSHADOW_XINLINE void Seed(mshadow::Stream<cpu> *, uint32_t seed) {
    for (int i = 0; i < kNumRandomStates; ++i) (states_ + i)->seed(seed + i);
  }

 private:
  std::mt19937 *states_;
};  // class RandGenerator<cpu, DType>

template<typename DType>
const int RandGenerator<cpu, DType>::kMinNumRandomPerThread = 64;

template<typename DType>
const int RandGenerator<cpu, DType>::kNumRandomStates = 1024;

#if MXNET_USE_CUDA

template<typename DType>
class RandGenerator<gpu, DType> {
 public:
  // at least how many random numbers should be generated by one GPU thread.
  static const int kMinNumRandomPerThread;
  // store how many global random states for GPU.
  static const int kNumRandomStates;

  // uniform number generation in Cuda made consistent with stl (include 0 but exclude 1)
  // by using 1.0-curand_uniform().
  // Needed as some samplers in sampler.h won't be able to deal with
  // one of the boundary cases.
  // TODO(alexzai): move impl class to separate file - tracked in MXNET-948
  class Impl {
   public:
    Impl &operator=(const Impl &) = delete;
    Impl(const Impl &) = delete;

    // Copy state to local memory for efficiency.
    __device__ explicit Impl(RandGenerator<gpu, DType> *gen, int state_idx)
        : global_gen_(gen),
          global_state_idx_(state_idx),
          state_(*(gen->states_ + state_idx)) {}

    __device__ ~Impl() {
      // store the curand state back into global memory
      global_gen_->states_[global_state_idx_] = state_;
    }

    MSHADOW_FORCE_INLINE __device__ int rand() {
      return curand(&state_);
    }

    MSHADOW_FORCE_INLINE __device__ float uniform() {
      return static_cast<float>(1.0) - curand_uniform(&state_);
    }

    MSHADOW_FORCE_INLINE __device__ float normal() {
      return curand_normal(&state_);
    }

   private:
    RandGenerator<gpu, DType> *global_gen_;
    int global_state_idx_;
    curandStatePhilox4_32_10_t state_;
  };  // class RandGenerator<gpu, DType>::Impl

  static void AllocState(RandGenerator<gpu, DType> *inst);

  static void FreeState(RandGenerator<gpu, DType> *inst);

  void Seed(mshadow::Stream<gpu> *s, uint32_t seed);

 private:
  curandStatePhilox4_32_10_t *states_;
};  // class RandGenerator<gpu, DType>

template<>
class RandGenerator<gpu, double> {
 public:
  // uniform number generation in Cuda made consistent with stl (include 0 but exclude 1)
  // by using 1.0-curand_uniform().
  // Needed as some samplers in sampler.h won't be able to deal with
  // one of the boundary cases.
  // TODO(alexzai): move impl class to separate file - tracked in MXNET-948
  class Impl {
   public:
    Impl &operator=(const Impl &) = delete;
    Impl(const Impl &) = delete;

    // Copy state to local memory for efficiency.
    __device__ explicit Impl(RandGenerator<gpu, double> *gen, int state_idx)
        : global_gen_(gen),
          global_state_idx_(state_idx),
          state_(*(gen->states_ + state_idx)) {}

    __device__ ~Impl() {
      // store the curand state back into global memory
      global_gen_->states_[global_state_idx_] = state_;
    }

    MSHADOW_FORCE_INLINE __device__ int rand() {
      return curand(&state_);
    }

    MSHADOW_FORCE_INLINE __device__ double uniform() {
      return static_cast<float>(1.0) - curand_uniform_double(&state_);
    }

    MSHADOW_FORCE_INLINE __device__ double normal() {
      return curand_normal_double(&state_);
    }

   private:
    RandGenerator<gpu, double> *global_gen_;
    int global_state_idx_;
    curandStatePhilox4_32_10_t state_;
  };  // class RandGenerator<gpu, double>::Impl

 private:
  curandStatePhilox4_32_10_t *states_;
};  // class RandGenerator<gpu, double>

#endif  // MXNET_USE_CUDA

}  // namespace random
}  // namespace common
}  // namespace mxnet
#endif  // MXNET_RANDOM_GENERATOR_H_
