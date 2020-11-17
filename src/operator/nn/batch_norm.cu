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
 * \file batch_norm.cu
 * \brief CUDA Batch Normalization code
 * \author Chris Olivier, Bing Xu, Da Zheng
 * Adapted from Torch
*/
#include <cuda_runtime_api.h>
#include <algorithm>
#include "batch_norm-inl.h"
#include "../../common/cuda/utils.h"


#define WRITE_DATA_FLAG       1
#define WRITE_GAMMA_FLAG      2
#define WRITE_BETA_FLAG       4
#define FIX_GAMMA_FLAG        8
#define IS_TRAINING_FLAG      16
#define USE_GLOBAL_STATS_FLAG 32
#define ADDTO_DATA_FLAG       (1 << 6)
#define ADDTO_GAMMA_FLAG      (1 << 7)
#define ADDTO_BETA_FLAG       (1 << 8)

#if MXNET_USE_CUDNN == 1
#include "./cudnn/cudnn_batch_norm-inl.h"
#endif

#include "../../../include/mxnet/tensor_blob.h"

using namespace mxnet;

namespace {

/*! \brief inverse standard deviation <-> variance */
template <typename DType, typename AccReal>
MSHADOW_XINLINE AccReal variance_to_invstd(DType var, AccReal eps) {
  return rsqrtf(static_cast<AccReal>(var) + eps);
}

template <>
MSHADOW_XINLINE double variance_to_invstd(double var, double eps) {
  return rsqrt(var + eps);
}

template <typename AccReal>
MSHADOW_XINLINE AccReal invstd_to_variance(AccReal invstd, AccReal eps) {
  return static_cast<AccReal>(1.0) / (invstd * invstd) - eps;
}

template <>
MSHADOW_XINLINE double invstd_to_variance(double invstd, double eps) {
  return 1.0 / (invstd * invstd) - eps;
}

}  // namespace

namespace mxnet {
namespace op {
namespace batchnorm {
namespace cuda {

static const unsigned WARP_SIZE = 32;

// The maximum number of threads in a block
static const unsigned MAX_BLOCK_SIZE = 512U;

template<typename In, typename Out>
struct ScalarConvert {
  static __host__ __device__ __forceinline__ Out to(const In v) { return (Out) v; }
};

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static unsigned getNumThreads(int nElem) {
  unsigned threadSizes[4] = {32, 64, 128, 256};
  for (int i = 0; i != 4; ++i) {
    if (static_cast<unsigned>(nElem) <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

template<typename DType, typename AccReal>
struct Float2 {
  AccReal v1, v2;
  __device__ Float2() {}
  __device__ Float2(DType v1, DType v2)
    : v1(ScalarConvert<DType, AccReal>::to(v1))
      , v2(ScalarConvert<DType, AccReal>::to(v2)) {}
  __device__ Float2(DType v)
    : v1(ScalarConvert<DType, AccReal>::to(v))
      , v2(ScalarConvert<DType, AccReal>::to(v)) {}
  __device__ Float2(int v)
    : v1(ScalarConvert<int, AccReal>::to(v))
      , v2(ScalarConvert<int, AccReal>::to(v)) {}
  __device__ Float2 &operator+=(const Float2 &a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template<typename DType, typename AccReal, typename DeviceTensor>
struct SumOp {
  __device__ SumOp(const DeviceTensor t) : tensor(t) {}
  __device__ __forceinline__ AccReal operator()(int batch, int plane, int n) {
    return ScalarConvert<DType, AccReal>::to(tensor.get_ref(batch, plane, n));
  }
  const DeviceTensor tensor;
};

template<typename DType, typename AccReal, typename DeviceTensor>
struct VarOp {
  __device__ VarOp(AccReal m, const DeviceTensor t)
    : mean(m)
      , tensor(t) {
  }
  __device__ __forceinline__ AccReal operator()(int batch, int plane, int n) {
    DType val = tensor.get_ref(batch, plane, n);
    return (val - mean) * (val - mean);
  }
  const AccReal mean;
  const DeviceTensor tensor;
};

template<typename DType, typename AccReal, typename DeviceTensor>
struct GradOp {
  __device__ GradOp(AccReal m, const DeviceTensor i, const DeviceTensor g)
    : mean(m), input(i), gradOutput(g) {}
  __device__ __forceinline__ Float2<DType, AccReal> operator()(int batch, int plane, int n) {
    const DType g = gradOutput.get_ref(batch, plane, n);
    const DType c = ScalarConvert<AccReal, DType>::to(input.get_ref(batch, plane, n) - mean);
    return Float2<DType, AccReal>(g, g * c);
  }
  const AccReal mean;
  const DeviceTensor input;
  const DeviceTensor gradOutput;
};

#if CUDA_VERSION >= 9000
#define FULLMASK 0xFFFFFFFF
#define __shfl_xor(...) __shfl_xor_sync(FULLMASK, __VA_ARGS__)
#endif

// Sum across all threads within a warp
template<typename T>
static __device__ __forceinline__ T warpSum(T val) {
#if __CUDA_ARCH__ >= 300
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += __shfl_xor(val, 1 << i, WARP_SIZE);
  }
#else
__shared__ T values[MAX_BLOCK_SIZE];
values[threadIdx.x] = val;
__threadfence_block();
const int base = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
for (int i = 1; i < WARP_SIZE; i++) {
val += values[base + ((i + threadIdx.x) % WARP_SIZE)];
}
#endif
return val;
}

template<typename DType, typename AccReal>
static __device__ __forceinline__ Float2<DType, AccReal> warpSum(Float2<DType, AccReal> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

// Sum across (batch, x/y/z) applying Op() pointwise
template<typename T, typename Op, typename DeviceTensor>
static __device__ T reduce(Op op, DeviceTensor tensor, int plane) {
  T sum = (T) 0;
  for (int batch = 0; batch < tensor.OuterSize(); ++batch) {
    for (int x = threadIdx.x; x < tensor.InnerSize(); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];
  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    shared[threadIdx.x / WARP_SIZE] = sum;
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (T) 0;
  }
  __syncthreads();
  if (threadIdx.x / WARP_SIZE == 0) {
    sum = warpSum(shared[threadIdx.x]);
    if (threadIdx.x == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

namespace {
  constexpr int inference_forward_threads = 512;
  constexpr int shmem_elements = 1536;
}  // namespace

template <typename DType, typename AType, typename LType, bool small_num_channels>
__launch_bounds__(inference_forward_threads)
__global__ void BatchNormalizationUpdateOutputInferenceKernel(
  const DType* input,
  DType* output,
  const index_t size,
  const index_t outer_size,
  const index_t num_channels,
  const index_t inner_size,
  const AType* runningMean,
  const AType* runningVar,
  AType* saveMean,
  AType* saveInvStd,
  AType* weight,
  AType* bias,
  const AType epsilon,
  const uint32_t flags) {
  constexpr int nvec = sizeof(LType) / sizeof(DType);
  __shared__ AType saved_invstd[shmem_elements];
  __shared__ AType saved_mean[shmem_elements];
  __shared__ AType saved_weight[shmem_elements];
  __shared__ AType saved_bias[shmem_elements];
  union vectorized_loader {
    LType aligned;
    DType separate[nvec];  // NOLINT(*)

    __device__ inline vectorized_loader() {}
    __device__ inline ~vectorized_loader() {}
  } scratch;

  if (small_num_channels) {
    for (int i = threadIdx.x; i < num_channels; i += blockDim.x) {
      saved_invstd[i] = variance_to_invstd(runningVar[i], epsilon);
      saved_mean[i] = runningMean[i];
      saved_weight[i] = (weight != nullptr && (flags & FIX_GAMMA_FLAG) == 0)
                        ? weight[i]
                        : 1;
      saved_bias[i] = (bias != nullptr) ? bias[i] : 0;
    }
    __syncthreads();
  }

  const index_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const index_t stride = blockDim.x * gridDim.x;
  const LType* input_aligned = reinterpret_cast<const LType*>(input);
  LType* output_aligned = reinterpret_cast<LType*>(output);
  for (index_t i = tid; i < size / nvec; i += stride) {
    scratch.aligned = input_aligned[i];
      const index_t my_channel_base = (nvec * i) % (inner_size * num_channels);
#pragma unroll
    for (int j = 0; j < nvec; ++j) {
      index_t my_channel = (my_channel_base + j) / inner_size;
      if (my_channel >= num_channels) my_channel = my_channel % num_channels;
      AType current_input = static_cast<AType>(scratch.separate[j]);

      AType invstd = small_num_channels ? saved_invstd[my_channel]
                                        : variance_to_invstd(runningVar[my_channel], epsilon);
      AType mean = small_num_channels ? saved_mean[my_channel]
                                      : runningMean[my_channel];
      AType gamma = small_num_channels ? saved_weight[my_channel]
                                       : ((weight != nullptr && (flags & FIX_GAMMA_FLAG) == 0)
                                          ? weight[my_channel]
                                          : 1);
      AType beta = small_num_channels ? saved_bias[my_channel]
                                      : ((bias != nullptr) ? bias[my_channel]
                                                           : 0);
      current_input = gamma * (current_input - mean) * invstd + beta;
      scratch.separate[j] = current_input;
    }

    output_aligned[i] = scratch.aligned;

    if (i < num_channels) {
      saveMean[i] = runningMean[i];
      saveInvStd[i] = variance_to_invstd(runningVar[i], epsilon);
      if ((flags & WRITE_GAMMA_FLAG) != 0 && (flags & FIX_GAMMA_FLAG) != 0
          && weight != nullptr) {
        weight[i] = 1;
      }
    }
  }
}

template<typename DType, typename AccReal, typename DeviceTensor1, typename DeviceTensor>
__global__ void BatchNormalizationUpdateOutputKernel(
  DeviceTensor input,
  DeviceTensor output,
  DeviceTensor1 weight,
  DeviceTensor1 bias,
  const AccReal epsilon,
  const AccReal momentum,
  DeviceTensor1 runningMean,
  DeviceTensor1 runningVar,
  DeviceTensor1 saveMean,
  DeviceTensor1 saveInvStd,
  const uint32_t flags) {
  const int plane = blockIdx.x;
  const int N = input.OuterSize() * input.InnerSize();

  const AccReal norm = AccReal(1) / N;

  // Compute the mean and variance across (batch, x/y/z)
  const AccReal mean = reduce<AccReal>(
    SumOp<DType, AccReal, DeviceTensor>(input), input, plane) * norm;
  __syncthreads();
  const AccReal varN = reduce<AccReal>(VarOp<DType, AccReal, DeviceTensor>(mean, input),
                                       input, plane);
  AccReal invStd = 0;
  if (varN != AccReal(0) || epsilon != AccReal(0)) {
    invStd = AccReal(1.0) / sqrt(varN * norm + epsilon);
  }

  // Save the mean, variance, and moving averages
  if (threadIdx.x == 0) {
    // For one item (0th) per plane (channel), write the per-channel data (ie mean, variance, etc)
    // Momentum based writeback
    saveMean[plane] = ScalarConvert<AccReal, DType>::to(mean);
    saveInvStd[plane] = invStd;
    if ((flags & WRITE_GAMMA_FLAG) != 0 && (flags & FIX_GAMMA_FLAG) != 0
        && weight.numElements() > 0) {
      weight[plane] = AccReal(1);
    }
  }

  // Write normalized and update the output
  const AccReal gamma = ((flags & FIX_GAMMA_FLAG) == 0 && weight.numElements() > 0)
                        ? ScalarConvert<DType, AccReal>::to(weight[plane])
                        : ScalarConvert<int, AccReal>::to(1);
  const AccReal beta = bias.numElements() > 0 ? ScalarConvert<DType, AccReal>::to(bias[plane])
                                              : ScalarConvert<int, AccReal>::to(0);
  for (int batch = 0, nbatch = input.OuterSize(); batch < nbatch; ++batch) {
    for (int x = threadIdx.x, nx = input.InnerSize(); x < nx; x += blockDim.x) {
      const DType inp = input.get_ref(batch, plane, x);
      output.get_ref(batch, plane, x) =
        ScalarConvert<AccReal, DType>::to(gamma * (inp - mean) * invStd + beta);
    }
  }
}

template<typename DeviceTensor1>
struct CUDATensors {
  DeviceTensor1 gradWeight;
  DeviceTensor1 gradBias;
  DeviceTensor1 weight;
  DeviceTensor1 runningMean;
  DeviceTensor1 runningVar;
  DeviceTensor1 saveMean;
  DeviceTensor1 saveInvStd;
};

namespace {
  inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
  }
}  // namespace

template<int NTHREADS, typename DType, typename AType, typename LType>
__global__ void FrozenBatchNormalizationBackwardKernelCLastPhase1(
    const DType* input, const DType* gradOutput, AType* temp_space,
    DType* gradInput, const AType* weight, const AType* runningMean,
    const AType* runningVar, const index_t outer, const index_t num_channels,
    const AType eps, const uint32_t flags) {
  using mxnet::common::cuda::warp_size;
  constexpr int num_warps = NTHREADS / warp_size;
  constexpr int nvec = sizeof(LType) >= sizeof(DType) ? sizeof(LType) / sizeof(DType) : 1;
  const size_t stride = num_channels / nvec;

  union vectorized_loader {
    LType aligned;
    DType separate[nvec];  // NOLINT(*)

    __device__ inline vectorized_loader() {}
    __device__ inline ~vectorized_loader() {}
  };

  vectorized_loader vec_input, vec_gradOutput;

  __shared__ AType scratch[NTHREADS * 2 * nvec];
  AType * my_values_gamma = &(scratch[threadIdx.x * nvec]);
  AType * my_values_beta = &(scratch[(NTHREADS + threadIdx.x) * nvec]);

  AType sum_gamma[nvec];  // NOLINT(*)
  AType sum_beta[nvec];  // NOLINT(*)
#pragma unroll
  for (int i = 0; i < nvec; ++i) {
    sum_gamma[i] = 0;
    sum_beta[i] = 0;
  }

  const size_t offset = blockIdx.x * warp_size;
  const int my_warp = threadIdx.x / warp_size;
  const int thread_idx_in_warp = threadIdx.x % warp_size;

  AType invstd[nvec];  // NOLINT(*)
  AType mean[nvec];  // NOLINT(*)
  AType gamma[nvec];  // NOLINT(*)
  size_t channel_offset = (offset + thread_idx_in_warp) * nvec;

  if (channel_offset < num_channels) {
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      invstd[i] = variance_to_invstd(runningVar[channel_offset + i], eps);
      mean[i] = runningMean[channel_offset + i];
      gamma[i] = weight != nullptr ? weight[channel_offset + i] : 1;
    }
  }

  const LType* aligned_gradOutput = reinterpret_cast<const LType*>(gradOutput);
  const LType* aligned_input = reinterpret_cast<const LType*>(input);
  LType* gradInput_aligned = reinterpret_cast<LType*>(gradInput);

  const int rows_per_block = (outer + gridDim.y - 1) / gridDim.y;
  const size_t start_row = my_warp + rows_per_block * blockIdx.y;
  const size_t end_row = min(outer, static_cast<index_t>(rows_per_block * (blockIdx.y + 1)));
  if (offset + thread_idx_in_warp < stride) {
    for (size_t i = start_row; i < end_row; i += num_warps) {
      const index_t idx = i * stride + offset + thread_idx_in_warp;
      vec_gradOutput.aligned = aligned_gradOutput[idx];
      vec_input.aligned = aligned_input[idx];
#pragma unroll
      for (int j = 0; j < nvec; ++j) {
        sum_beta[j]  += static_cast<AType>(vec_gradOutput.separate[j]);
        sum_gamma[j] += static_cast<AType>(vec_gradOutput.separate[j]) *
                        (static_cast<AType>(vec_input.separate[j]) - mean[j]);
      }
      if (flags & (WRITE_DATA_FLAG | ADDTO_DATA_FLAG)) {
        // Gradient to input
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          vec_gradOutput.separate[j] *= invstd[j] * gamma[j];
        }
        if (flags & ADDTO_DATA_FLAG) {
           vec_input.aligned = gradInput_aligned[idx];
#pragma unroll
           for (int j = 0; j < nvec; ++j) {
             vec_gradOutput.separate[j] += vec_input.separate[j];
           }
        }
        gradInput_aligned[idx] = vec_gradOutput.aligned;
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < nvec; ++i) {
    my_values_gamma[i] = sum_gamma[i];
    my_values_beta[i] = sum_beta[i];
  }

  __syncthreads();

  for (int i = num_warps / 2; i > 0; i /= 2) {
    if (my_warp < i) {
      const int shared_offset = nvec * i * warp_size;
#pragma unroll
      for (int j = 0; j < nvec; ++j) {
        my_values_gamma[j] += my_values_gamma[j + shared_offset];
        my_values_beta[j] += my_values_beta[j + shared_offset];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x < min(warp_size * nvec,
                        static_cast<int>(num_channels - nvec * offset))) {
    const size_t offset_out = nvec * offset +
                              blockIdx.y * num_channels;
    const size_t offset_beta = gridDim.y * num_channels;
    temp_space[offset_out + threadIdx.x] = scratch[threadIdx.x];
    temp_space[offset_beta + offset_out + threadIdx.x] = scratch[NTHREADS * nvec + threadIdx.x];
  }
}

template <typename AType>
__global__ void FrozenBatchNormalizationBackwardKernelCLastPhase2(const AType * temp_space,
                                                                  const AType * runningVar,
                                                                  AType * out_gamma,
                                                                  AType * out_beta,
                                                                  int lead_dim, int n_blocks,
                                                                  AType epsilon, uint32_t flags) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < lead_dim) {
    AType sum_gamma = 0;
    AType sum_beta = 0;
    for (int i = tid; i < lead_dim * n_blocks; i += lead_dim) {
      sum_gamma += temp_space[i];
      sum_beta += temp_space[i + lead_dim * n_blocks];
    }
    if (flags & (WRITE_GAMMA_FLAG | ADDTO_GAMMA_FLAG)) {
      if ((flags & FIX_GAMMA_FLAG) == 0) {
        const AType invstd = variance_to_invstd(runningVar[tid], epsilon);
        if (flags & WRITE_GAMMA_FLAG) {
          out_gamma[tid] = sum_gamma * invstd;
        } else {
          out_gamma[tid] += sum_gamma * invstd;
        }
      } else {
        if (flags & WRITE_GAMMA_FLAG) {
          out_gamma[tid] = 0;
        }
      }
    }
    if (flags & WRITE_BETA_FLAG) {
      out_beta[tid] = sum_beta;
    } else if (flags & ADDTO_BETA_FLAG) {
      out_beta[tid] += sum_beta;
    }
  }
}

template<int NTHREADS, typename DType, typename AType, typename LType>
__global__ void FrozenBatchNormalizationBackwardKernel(
  const DType* input,
  const DType* gradOutput,
  DType* gradInput,
  AType* gradWeight,
  AType* gradBias,
  const AType* weight,
  const AType* runningMean,
  const AType* runningVar,
  const index_t outer,
  const index_t inner,
  const index_t num_channels,
  const index_t NHW_div_nvec,
  const AType eps,
  const uint32_t flags) {
  const index_t my_channel = blockIdx.x;
  const AType invstd = variance_to_invstd(runningVar[my_channel], eps);
  const AType mean = runningMean[my_channel];
  const AType gamma = weight != nullptr ? weight[my_channel] : 1;
  constexpr int nvec = sizeof(LType) > sizeof(DType) ? sizeof(LType) / sizeof(DType)
                                                     : 1;
  union vectorized_loader {
    LType aligned;
    DType separate[nvec];  // NOLINT(*)

    __device__ inline vectorized_loader() {}
    __device__ inline ~vectorized_loader() {}
  };

  vectorized_loader vec_input, vec_gradOutput;

  const LType* input_aligned = reinterpret_cast<const LType*>(input);
  const LType* gradOutput_aligned = reinterpret_cast<const LType*>(gradOutput);
  LType* gradInput_aligned = reinterpret_cast<LType*>(gradInput);

  const index_t inner_div_nvec = inner / nvec;

  AType sum_gamma = 0;
  AType sum_beta = 0;


  for (index_t i = threadIdx.x; i < NHW_div_nvec; i += blockDim.x) {
    const index_t inner_idx = i % inner_div_nvec;
    const index_t outer_idx = i / inner_div_nvec;
    const index_t idx = inner_idx +
                        (my_channel + outer_idx * num_channels) * inner_div_nvec;
    vec_gradOutput.aligned = gradOutput_aligned[idx];
    vec_input.aligned = input_aligned[idx];
#pragma unroll
    for (int j = 0; j < nvec; ++j) {
      sum_beta += static_cast<AType>(vec_gradOutput.separate[j]);
      sum_gamma += static_cast<AType>(vec_gradOutput.separate[j]) *
                   (static_cast<AType>(vec_input.separate[j]) - mean);
    }

    if (flags & (WRITE_DATA_FLAG | ADDTO_DATA_FLAG)) {
      // Gradient to input
#pragma unroll
      for (int j = 0; j < nvec; ++j) {
        vec_gradOutput.separate[j] *= invstd * gamma;
      }
      if (flags & ADDTO_DATA_FLAG) {
         vec_input.aligned = gradInput_aligned[idx];
#pragma unroll
         for (int j = 0; j < nvec; ++j) {
           vec_gradOutput.separate[j] += vec_input.separate[j];
         }
      }
      gradInput_aligned[idx] = vec_gradOutput.aligned;
    }
  }

  sum_gamma = common::cuda::reduce<NTHREADS, false>(sum_gamma,
                                                    [](AType a, AType b) { return a + b; });
  sum_beta = common::cuda::reduce<NTHREADS, false>(sum_beta,
                                                   [](AType a, AType b) { return a + b; });

  if (threadIdx.x == 0) {
    if (flags & (WRITE_GAMMA_FLAG | ADDTO_GAMMA_FLAG)) {
      if ((flags & FIX_GAMMA_FLAG) == 0) {
        if (flags & WRITE_GAMMA_FLAG) {
          gradWeight[my_channel] = sum_gamma * invstd;
        } else {
          gradWeight[my_channel] += sum_gamma * invstd;
        }
      } else {
        if (flags & WRITE_GAMMA_FLAG) {
          gradWeight[my_channel] = 0;
        }
      }
    }
    if (flags & WRITE_BETA_FLAG) {
      gradBias[my_channel] = sum_beta;
    } else if (flags & ADDTO_BETA_FLAG) {
      gradBias[my_channel] += sum_beta;
    }
  }
}

template<typename DType, typename AccReal, typename DeviceTensor1, typename DeviceTensor>
static __global__ void BatchNormalizationBackwardKernel(
  const DeviceTensor input,
  const DeviceTensor gradOutput,
  DeviceTensor gradInput,
  CUDATensors<DeviceTensor1> tensors,
  const uint32_t flags,
  const AccReal momentum,
  const AccReal eps) {
  int plane = blockIdx.x;
  int N = gradOutput.OuterSize() * gradOutput.InnerSize();

  AccReal mean, invstd;
  mean = ScalarConvert<DType, AccReal>::to(tensors.saveMean[plane]);
  invstd = tensors.saveInvStd[plane];

  const AccReal weightVal = ((flags & FIX_GAMMA_FLAG) == 0 && tensors.weight.numElements() > 0) ?
                      ScalarConvert<DType, AccReal>::to(tensors.weight[plane]) : AccReal(1);
  const AccReal norm = AccReal(1) / N;

  // Compute two values across (batch, x/y/z) in one pass:
  // 1. Sum(gradOutput)
  // 2. DotProduct(input - mean, gradOutput)
  GradOp<DType, AccReal, DeviceTensor> g(mean, input, gradOutput);
  Float2< DType, AccReal > res = reduce < Float2 < DType, AccReal >,
    GradOp< DType, AccReal, DeviceTensor >, DeviceTensor > (g, gradOutput, plane);
  const AccReal gradOutputSum = res.v1;
  const AccReal dotP = res.v2;

  const AccReal gradMean = gradOutputSum * norm;
  const AccReal projScale = dotP * norm * invstd * invstd;
  const AccReal gradScale = invstd * weightVal;

  if (threadIdx.x == 0) {
    const AccReal localVariance = invstd_to_variance(tensors.saveInvStd[plane], eps);
    const AccReal localMean = tensors.saveMean[plane];

    // update running averages
    tensors.runningMean[plane] = tensors.runningMean[plane]
                                 * momentum + localMean * (AccReal(1) - momentum);
    tensors.runningVar[plane] = tensors.runningVar[plane]
                                * momentum + localVariance * (AccReal(1) - momentum);
  }

  if (gradInput.Size() > 0 && (flags & (WRITE_DATA_FLAG | ADDTO_DATA_FLAG)) != 0) {
    const bool grad_write = flags & WRITE_DATA_FLAG;
    if (grad_write) {
      for (int batch = 0, nbatch = gradOutput.OuterSize(); batch < nbatch; ++batch) {
        for (int x = threadIdx.x, nx = gradOutput.InnerSize(); x < nx; x += blockDim.x) {
          const DType gradOut = gradOutput.get_ref(batch, plane, x);
          const DType inp = input.get_ref(batch, plane, x);
          const AccReal proj = (inp - mean) * projScale;
          gradInput.get_ref(batch, plane, x) =
            ScalarConvert<AccReal, DType>::to((gradOut - proj - gradMean) * gradScale);
        }
      }
    } else {
      // grad addto
      for (int batch = 0, nbatch = gradOutput.OuterSize(); batch < nbatch; ++batch) {
        for (int x = threadIdx.x, nx = gradOutput.InnerSize(); x < nx; x += blockDim.x) {
          const DType gradOut = gradOutput.get_ref(batch, plane, x);
          const DType inp = input.get_ref(batch, plane, x);
          const AccReal proj = (inp - mean) * projScale;
          gradInput.get_ref(batch, plane, x) +=
            ScalarConvert<AccReal, DType>::to((gradOut - proj - gradMean) * gradScale);
        }
      }
    }
  }

  if (tensors.gradWeight.numElements() > 0 && threadIdx.x == 0 &&
      (flags & (WRITE_GAMMA_FLAG | ADDTO_GAMMA_FLAG)) != 0) {
    if ((flags & FIX_GAMMA_FLAG) == 0) {
      if (flags & WRITE_GAMMA_FLAG)
        tensors.gradWeight[plane] = ScalarConvert<AccReal, DType>::to(dotP * invstd);
      else
        tensors.gradWeight[plane] += ScalarConvert<AccReal, DType>::to(dotP * invstd);
    } else {
      tensors.gradWeight[plane] = DType(0);
    }
  }

  if (tensors.gradBias.numElements() > 0 && threadIdx.x == 0 &&
      (flags & (WRITE_BETA_FLAG | ADDTO_BETA_FLAG)) != 0) {
    if (flags & WRITE_BETA_FLAG)
      tensors.gradBias[plane] = ScalarConvert<AccReal, DType>::to(gradOutputSum);
    else
      tensors.gradBias[plane] += ScalarConvert<AccReal, DType>::to(gradOutputSum);
  }
}

template<typename DType, int Dim>
struct DeviceTensor {
 public:
  inline DeviceTensor() {}
  inline DeviceTensor(DType *p, const int *size)
    : dptr_(p) {
    for (int i = 0; i < Dim; ++i) {
      size_[i] = size ? size[i] : 0;
    }
  }

  MSHADOW_XINLINE unsigned getSize(const int i) const {
    return size_[i];
  }

  MSHADOW_XINLINE int numElements() const {
    int n = 1;
    for (int i = 0; i < Dim; ++i) {
      n *= size_[i];
    }
    return n;
  }

  MSHADOW_XINLINE DType &operator()(const size_t batch,
                                    const size_t plane,
                                    const size_t x) const {
    int offset = 0;

    offset *= size_[0];
    offset += batch;

    offset *= size_[1];
    offset += plane;

    offset *= size_[2];
    offset += x;

    return *(const_cast<DType *>(dptr_ + offset));
  }

  MSHADOW_XINLINE DType &operator[](const size_t x) const {
    return *(dptr_ + x);
  }

  MSHADOW_XINLINE size_t InnerSize() const {
    size_t sz = 1;
    for (size_t i = 2; i < Dim; ++i) {
      sz *= size_[i];
    }
    return sz;
  }

  MSHADOW_XINLINE size_t ChannelCount() const {
    return size_[1];
  }

  DType *dptr_;
  int size_[Dim];
};

template<typename DType, int Dim>
static DeviceTensor<DType, Dim> devicetensor(const TBlob &blob) {
  CHECK_EQ(blob.type_flag_, mshadow::DataType<DType>::kFlag);
  DType *data = blob.dptr<DType>();
  const int inDim = blob.shape_.ndim();
  if (inDim == Dim) {
    DeviceTensor<DType, Dim> tensor(data, nullptr);
    for (int i = 0; i < Dim; ++i) {
      tensor.size_[i] = blob.size(i);
    }
    return tensor;
  }

  // View in which the last dimensions are collapsed or expanded as needed
  int size[Dim];
  for (int i = 0; i < Dim || i < inDim; ++i) {
    if (i < Dim && i < inDim) {
      size[i] = blob.size(i);
    } else if (i < Dim) {
      size[i] = 1;
    } else {
      size[Dim - 1] *= blob.size(i);
    }
  }
  return DeviceTensor<DType, Dim>(data, &size[0]);
}


#define DeviceTensor1 DeviceTensor<AccReal, 1>

using namespace mxnet::op;

template<typename DType, typename AccReal>
static void BatchNormalizationUpdateOutput(mshadow::Stream<gpu> *s,
                                           const OpContext &ctx,
                                           const BatchNormParam& param,
                                           const std::vector<TBlob> &in_data,
                                           const std::vector<TBlob> &out_data,
                                           const std::vector<TBlob> &aux_states,
                                           const uint32_t flags,
                                           double momentum,
                                           double eps) {
  batchnorm::BNTensor3<DType> input  = batchnorm::BNTensor3<DType>(
    in_data[batchnorm::kData], param.axis);
  batchnorm::BNTensor3<DType> output = batchnorm::BNTensor3<DType>(
    out_data[batchnorm::kOut], param.axis);
  DeviceTensor1 weight = devicetensor<AccReal, 1>(in_data[batchnorm::kGamma]);
  DeviceTensor1 bias = devicetensor<AccReal, 1>(in_data[batchnorm::kBeta]);
  DeviceTensor1 runningMean = devicetensor<AccReal, 1>(aux_states[batchnorm::kMovingMean]);
  DeviceTensor1 runningVar = devicetensor<AccReal, 1>(aux_states[batchnorm::kMovingVar]);
  DeviceTensor1 saveMean = devicetensor<AccReal, 1>(out_data[batchnorm::kMean]);
  DeviceTensor1 saveInvStd = devicetensor<AccReal, 1>(out_data[batchnorm::kVar]);

  DCHECK_GT(weight.numElements(), 0);

  if ((flags & IS_TRAINING_FLAG) == 0 || (flags & USE_GLOBAL_STATS_FLAG) != 0) {
    AccReal* bias_ptr = bias.numElements() > 0 ? bias.dptr_ : nullptr;
    AccReal* gamma_ptr = weight.numElements() > 0 ? weight.dptr_ : nullptr;
    int nvec = sizeof(double) / sizeof(DType);
    index_t size = input.InnerSize() * input.OuterSize() * input.ChannelCount();
    index_t aligned_size = ((size + nvec - 1) / nvec) * nvec;
    index_t blocks = std::min((size + nvec * inference_forward_threads - 1) /
                              (nvec * inference_forward_threads),
                              static_cast<index_t>(512));
    if (input.ChannelCount() < shmem_elements) {
      BatchNormalizationUpdateOutputInferenceKernel<DType, AccReal, double, true>
        <<<blocks, inference_forward_threads, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
        input.dptr_, output.dptr_,
        aligned_size, input.OuterSize(),
        input.ChannelCount(), input.InnerSize(),
        runningMean.dptr_, runningVar.dptr_,
        saveMean.dptr_, saveInvStd.dptr_,
        gamma_ptr, bias_ptr,
        eps, flags);
    } else {
      BatchNormalizationUpdateOutputInferenceKernel<DType, AccReal, double, false>
        <<<blocks, inference_forward_threads, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
        input.dptr_, output.dptr_,
        aligned_size, input.OuterSize(),
        input.ChannelCount(), input.InnerSize(),
        runningMean.dptr_, runningVar.dptr_,
        saveMean.dptr_, saveInvStd.dptr_,
        gamma_ptr, bias_ptr,
        eps, flags);
    }
  } else {
    dim3 blocks(input.ChannelCount());
    dim3 threads(batchnorm::cuda::getNumThreads(input.InnerSize()));
    BatchNormalizationUpdateOutputKernel<DType, AccReal, DeviceTensor1,
      batchnorm::BNTensor3<DType>>
      << < blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s) >> > (
      input, output, weight, bias, eps, momentum, runningMean, runningVar,
        saveMean, saveInvStd, flags);
  }
  MSHADOW_CUDA_POST_KERNEL_CHECK(BatchNormalizationUpdateOutput);
}

template<typename DType, typename AccReal>
static void BatchNormalizationBackward(mshadow::Stream<gpu> *s,
                                       const OpContext &ctx,
                                       const BatchNormParam& param,
                                       const std::vector<TBlob> &out_grad,
                                       const std::vector<TBlob> &in_data,
                                       const std::vector<TBlob> &out_data,
                                       const std::vector<TBlob> &in_grad,
                                       const std::vector<TBlob> &aux_states,
                                       const uint32_t flags,
                                       double momentum,
                                       double eps) {
  batchnorm::BNTensor3<DType> input = batchnorm::BNTensor3<DType>(
    in_data[batchnorm::kData], param.axis);
  batchnorm::BNTensor3<DType>gradOutput = batchnorm::BNTensor3<DType>(
    out_grad[batchnorm::kOut], param.axis);
  batchnorm::BNTensor3<DType>gradInput = batchnorm::BNTensor3<DType>(
    in_grad[batchnorm::kData], param.axis);

  CHECK_EQ(gradOutput.Size(), gradInput.Size());

  CUDATensors<DeviceTensor1> tensors;

  tensors.gradWeight = devicetensor<AccReal, 1>(in_grad[batchnorm::kGamma]);
  tensors.gradBias = devicetensor<AccReal, 1>(in_grad[batchnorm::kBeta]);
  tensors.weight = devicetensor<AccReal, 1>(in_data[batchnorm::kGamma]);
  tensors.runningMean = devicetensor<AccReal, 1>(aux_states[batchnorm::kMovingMean]);
  tensors.runningVar = devicetensor<AccReal, 1>(aux_states[batchnorm::kMovingVar]);
  tensors.saveMean = devicetensor<AccReal, 1>(out_data[batchnorm::kMean]);
  tensors.saveInvStd = devicetensor<AccReal, 1>(out_data[batchnorm::kVar]);

  DCHECK_GT(tensors.weight.numElements(), 0);
  const bool is_train_and_not_global_stats =
    (flags & IS_TRAINING_FLAG) != 0 && (flags & USE_GLOBAL_STATS_FLAG) == 0;

  if (is_train_and_not_global_stats) {
#ifdef NDEBUG
    constexpr bool SMALLER_THREADS = false;
#else
    constexpr bool SMALLER_THREADS = true;
#endif
    dim3 blocks(gradOutput.ChannelCount());
    dim3 threads(batchnorm::cuda::getNumThreads(gradOutput.InnerSize()));
    BatchNormalizationBackwardKernel<DType, AccReal, DeviceTensor1, batchnorm::BNTensor3<DType>>
      <<< blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s) >>> (
      input, gradOutput, gradInput, tensors, flags, momentum, eps);
  } else {
    uint32_t flags_copy = flags;
    if (gradInput.Size() <= 0) {
      flags_copy = (flags_copy & ~WRITE_DATA_FLAG);
    }
    if (tensors.gradWeight.numElements() <= 0) {
      flags_copy = (flags_copy & ~WRITE_GAMMA_FLAG);
    }
    if (tensors.gradBias.numElements() <= 0) {
      flags_copy = (flags_copy & ~WRITE_BETA_FLAG);
    }
    AccReal* gamma = ((flags & FIX_GAMMA_FLAG) == 0 && tensors.weight.numElements() > 0)
                     ? tensors.weight.dptr_
                     : nullptr;

    if (param.axis == -1 || param.axis == in_data[batchnorm::kData].shape_.ndim() - 1) {
      const int C = gradOutput.ChannelCount();
      int ltype = mxnet::common::cuda::get_load_type(C * sizeof(DType));
      const int M = gradOutput.OuterSize();
      MXNET_LOAD_TYPE_SWITCH(ltype, LType, {
        const unsigned int blocks_x = ceil_div(C * sizeof(DType),
                                               mxnet::common::cuda::warp_size * sizeof(LType));
        const unsigned int preferred_number_of_blocks = 2 *
                                                        MultiprocessorCount(ctx.run_ctx.ctx.dev_id);
        const unsigned int blocks_y = std::max(preferred_number_of_blocks / blocks_x, 1u);
        const dim3 n_blocks = {blocks_x, blocks_y, 1};
        auto scratch_space = ctx.requested[batchnorm::kTempSpace]
                                .get_space_typed<gpu, 1, AccReal>(mshadow::Shape1(C * blocks_y * 2),
                                                                                  s);
        auto stream = mshadow::Stream<gpu>::GetStream(s);
        constexpr int nthreads_phase1 = 512;
        constexpr int nthreads_phase2 = 128;
        FrozenBatchNormalizationBackwardKernelCLastPhase1<nthreads_phase1, DType, AccReal, LType>
          <<<n_blocks, nthreads_phase1, 0, stream>>>(input.dptr_, gradOutput.dptr_,
                                                     scratch_space.dptr_,
                                                     gradInput.dptr_,
                                                     gamma,
                                                     tensors.runningMean.dptr_,
                                                     tensors.runningVar.dptr_,
                                                     M, C, eps, flags_copy);
        const int nblocks_phase2 = ceil_div(C, nthreads_phase2);
        FrozenBatchNormalizationBackwardKernelCLastPhase2<AccReal>
          <<<nblocks_phase2, nthreads_phase2, 0, stream>>>(scratch_space.dptr_,
                                                           tensors.runningVar.dptr_,
                                                           tensors.gradWeight.dptr_,
                                                           tensors.gradBias.dptr_, C,
                                                           blocks_y, eps, flags_copy);
      });
    } else {
    dim3 blocks(gradOutput.ChannelCount());
    int ltype = mxnet::common::cuda::get_load_type(gradOutput.InnerSize() * sizeof(DType));
    MXNET_LOAD_TYPE_SWITCH(ltype, LType, {
      constexpr int nvec = sizeof(LType) > sizeof(DType) ? sizeof(LType) / sizeof(DType) : 1;
      const index_t NHW_div_nvec = gradOutput.OuterSize() * gradOutput.InnerSize() / nvec;
      constexpr int threads = 512;
      FrozenBatchNormalizationBackwardKernel<threads, DType, AccReal, LType>
        <<< blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s) >>> (
        input.dptr_, gradOutput.dptr_, gradInput.dptr_,
        tensors.gradWeight.dptr_, tensors.gradBias.dptr_,
        gamma, tensors.runningMean.dptr_,
        tensors.runningVar.dptr_,
        gradOutput.OuterSize(), gradOutput.InnerSize(),
        gradOutput.ChannelCount(), NHW_div_nvec, eps, flags_copy);
      });
    }
  }
  MSHADOW_CUDA_POST_KERNEL_CHECK(BatchNormalizationBackward);
}

}  // namespace cuda
}  // namespace batchnorm

template<typename xpu, typename DType, typename AccReal>
static inline uint32_t SetupFlags(const OpContext &ctx,
                                  const BatchNormParam& params,
                                  const std::vector<OpReqType> &req) {
  uint32_t flags = 0;
  flags |= ctx.is_train ? IS_TRAINING_FLAG : 0;
  flags |= params.fix_gamma ? FIX_GAMMA_FLAG : 0;
  flags |= params.use_global_stats ? USE_GLOBAL_STATS_FLAG : 0;
  if (IsBNWriting(req[batchnorm::kData])) {
    flags |= WRITE_DATA_FLAG;
  } else if (req[batchnorm::kData] == kAddTo) {
    flags |= ADDTO_DATA_FLAG;
  }
  if (IsBNWriting(req[batchnorm::kGamma])) {
    flags |= WRITE_GAMMA_FLAG;
  } else if (req[batchnorm::kGamma] == kAddTo) {
    flags |= ADDTO_GAMMA_FLAG;
  }
  if (IsBNWriting(req[batchnorm::kBeta])) {
    flags |= WRITE_BETA_FLAG;
  } else if (req[batchnorm::kBeta] == kAddTo) {
    flags |= ADDTO_BETA_FLAG;
  }
  return flags;
}

/*! \brief Forward batch-norm pass on GPU */
template<typename xpu, typename DType, typename AccReal>
void BatchNormForwardImpl(mshadow::Stream<gpu> *stream,
                          const OpContext &ctx, const BatchNormParam& param_,
                          const std::vector<TBlob> &in_data,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &out_data,
                          const std::vector<TBlob> &aux_states) {
  batchnorm::cuda::BatchNormalizationUpdateOutput<DType, AccReal>(
    stream,
    ctx,
    param_,
    in_data,
    out_data,
    aux_states,
    SetupFlags<xpu, DType, AccReal>(ctx, param_, req),
    param_.momentum,
    param_.eps);
  MSHADOW_CUDA_POST_KERNEL_CHECK(BatchNormOp_DoForward_gpu);
}

/*! \brief Backward batch-norm pass on GPU */
template<typename xpu, typename DType, typename AccReal>
void BatchNormBackwardImpl(mshadow::Stream<gpu> *stream,
                           const OpContext &ctx, const BatchNormParam& param_,
                           const std::vector<TBlob> &out_grad,
                           const std::vector<TBlob> &in_data,
                           const std::vector<TBlob> &out_data,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &in_grad,
                           const std::vector<TBlob> &aux_states) {
  batchnorm::cuda::BatchNormalizationBackward<DType, AccReal>(
    stream,
    ctx,
    param_,
    out_grad,
    in_data,
    out_data,
    in_grad,
    aux_states,
    SetupFlags<xpu, DType, AccReal>(ctx, param_, req),
    param_.momentum,
    param_.eps);
  MSHADOW_CUDA_POST_KERNEL_CHECK(BatchNormOp_DoBackward_gpu);
}

#if MXNET_USE_CUDNN == 1
template<typename DType>
static CuDNNBatchNormOp<DType> &GetCuDNNOp(const BatchNormParam& param) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local CuDNNBatchNormOp<DType> op;
#else
  static MX_THREAD_LOCAL CuDNNBatchNormOp<DType> op;
#endif
  op.Init(param);
  return op;
}
#endif

template<>
void BatchNormCompute<gpu>(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx, const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  BatchNormParam param = nnvm::get<BatchNormParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 5U);
  std::vector<TBlob> in_data(inputs.begin(), inputs.begin() + 3);
  std::vector<TBlob> aux_states(inputs.begin() + 3, inputs.end());
  int dtype = inputs[0].type_flag_;
  mxnet::TShape shape = inputs[0].shape_;

  param.axis = mxnet::op::batchnorm::GetRealAxis(shape, param.axis);
#if MXNET_USE_CUDNN == 1
  if (!param.use_global_stats && !param.cudnn_off) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      GetCuDNNOp<DType>(param).Forward(ctx, in_data, req, outputs, aux_states);
    })
  } else {
    MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DType, AccReal, {
      BatchNormForward<gpu, DType, AccReal>(ctx, param, in_data, req, outputs, aux_states);
    })
  }
#else
  MSHADOW_REAL_TYPE_SWITCH_EX(inputs[0].type_flag_, DType, AccReal, {
    BatchNormForward<gpu, DType, AccReal>(ctx, param, in_data, req, outputs, aux_states);
  });
#endif
}

template<>
void BatchNormGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx, const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 8U);
  BatchNormParam param = nnvm::get<BatchNormParam>(attrs.parsed);
  int dtype = inputs[0].type_flag_;
  mxnet::TShape shape = inputs[0].shape_;

  param.axis = mxnet::op::batchnorm::GetRealAxis(shape, param.axis);
#if MXNET_USE_CUDNN == 1
  if (!param.use_global_stats && !param.cudnn_off) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      GetCuDNNOp<DType>(param).Backward(ctx, inputs, req, outputs);
    })
  } else {
    MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DType, AccReal, {
      BatchNormBackward<gpu, DType, AccReal>(ctx, param, inputs, req, outputs);
    })
  }
#else
  MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DType, AccReal, {
    BatchNormBackward<gpu, DType, AccReal>(ctx, param, inputs, req, outputs);
  });
#endif
}

NNVM_REGISTER_OP(BatchNorm)
.set_attr<FCompute>("FCompute<gpu>", BatchNormCompute<gpu>);

NNVM_REGISTER_OP(_backward_BatchNorm)
.set_attr<FCompute>("FCompute<gpu>", BatchNormGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet
