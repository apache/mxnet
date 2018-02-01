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
 * \author Chris Olivier, Bing Xu
 * Adapted from Torch
*/
#include <cuda_runtime_api.h>
#include <algorithm>
#include "batch_norm-inl.h"

#define WRITE_DATA_FLAG       1
#define WRITE_GAMMA_FLAG      2
#define WRITE_BETA_FLAG       4
#define FIX_GAMMA_FLAG        8
#define IS_TRAINING_FLAG      16
#define USE_GLOBAL_STATS_FLAG 32

#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
#include "./cudnn/cudnn_batch_norm-inl.h"
#endif

#include "../../common/cuda_utils.h"

using namespace mxnet;

/*! \brief inverse standard deviation <-> variance */
#define VARIANCE_TO_INVSTD(__var$,    __eps$)   (1.0/sqrt((__var$) + DType(__eps$)))
#define INVSTD_TO_VARIANCE(__invstd$, __eps$)   ((1.0 / ((__invstd$) * (__invstd$))) - (__eps$))

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
static unsigned getNumThreads(int nElem, const bool smaller) {
  unsigned threadSizes[5] = {32, 64, 128, 256, MAX_BLOCK_SIZE};
  const int maxi = smaller ? 4 : 5;
  for (int i = 0; i != maxi; ++i) {
    if (static_cast<unsigned>(nElem) <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return smaller ? (MAX_BLOCK_SIZE >> 1) : MAX_BLOCK_SIZE;
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

template <typename DType, typename AccReal, typename DeviceTensor1, typename DeviceTensor>
__global__ void BatchNormalizationUpdateOutputInferenceKernel(
  DeviceTensor input,
  DeviceTensor output,
  DeviceTensor1 runningMean,
  DeviceTensor1 runningVar,
  DeviceTensor1 saveMean,
  DeviceTensor1 saveInvStd,
  DeviceTensor1 weight,
  DeviceTensor1 bias,
  const DType epsilon,
  const uint32_t flags) {
  int plane = blockIdx.x;

  AccReal invstd = VARIANCE_TO_INVSTD(runningVar[plane], epsilon);
  AccReal mean = ScalarConvert<DType, AccReal>::to(runningMean[plane]);
  AccReal gamma = ((flags & FIX_GAMMA_FLAG) == 0 && weight.numElements() > 0)
                  ? ScalarConvert<DType, AccReal>::to(weight[plane])
                  : ScalarConvert<int, AccReal>::to(1);
  AccReal beta = bias.numElements() > 0 ? ScalarConvert<DType, AccReal>::to(bias[plane])
                                        : ScalarConvert<int, AccReal>::to(0);
  if (threadIdx.x == 0) {
    saveMean[plane] = runningMean[plane];
    saveInvStd[plane] = VARIANCE_TO_INVSTD(runningVar[plane], epsilon);
    if ((flags & WRITE_GAMMA_FLAG) != 0 && (flags & FIX_GAMMA_FLAG) != 0
        && weight.numElements() > 0) {
      weight[plane] = AccReal(1);
    }
  }
  // Write normalized and update the output
  for (int batch = 0, nbatch = input.OuterSize(); batch < nbatch; ++batch) {
    for (int x = threadIdx.x, nx = input.InnerSize(); x < nx; x += blockDim.x) {
      const DType inp = input.get_ref(batch, plane, x);
      output.get_ref(batch, plane, x) =
        ScalarConvert<AccReal, DType>::to(gamma * (inp - mean) * invstd + beta);
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

template<typename DType, typename AccReal, typename DeviceTensor1, typename DeviceTensor>
static __global__ void BatchNormalizationBackwardKernel(
  const DeviceTensor input,
  const DeviceTensor gradOutput,
  DeviceTensor gradInput,
  CUDATensors<DeviceTensor1> tensors,
  const uint32_t flags,
  const AccReal momentum,
  const double eps) {
  int plane = blockIdx.x;
  int N = gradOutput.OuterSize() * gradOutput.InnerSize();

  const bool is_train_and_not_global_stats =
    (flags & IS_TRAINING_FLAG) != 0 && (flags & USE_GLOBAL_STATS_FLAG) == 0;

  AccReal mean, invstd;
  if (is_train_and_not_global_stats) {
    mean = ScalarConvert<DType, AccReal>::to(tensors.saveMean[plane]);
    invstd = tensors.saveInvStd[plane];
  } else {
    mean = ScalarConvert<DType, AccReal>::to(tensors.runningMean[plane]);
    invstd = VARIANCE_TO_INVSTD(tensors.runningVar[plane], eps);
  }

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

  if (threadIdx.x == 0 && is_train_and_not_global_stats) {
    const AccReal localVariance = INVSTD_TO_VARIANCE(tensors.saveInvStd[plane], eps);
    const AccReal localMean = tensors.saveMean[plane];

    // update running averages
    tensors.runningMean[plane] = tensors.runningMean[plane]
                                 * momentum + localMean * (AccReal(1) - momentum);
    tensors.runningVar[plane] = tensors.runningVar[plane]
                                * momentum + localVariance * (AccReal(1) - momentum);
  }

  if (gradInput.Size() > 0 && (flags & WRITE_DATA_FLAG) != 0) {
    for (int batch = 0, nbatch = gradOutput.OuterSize(); batch < nbatch; ++batch) {
      for (int x = threadIdx.x, nx = gradOutput.InnerSize(); x < nx; x += blockDim.x) {
        const DType gradOut = gradOutput.get_ref(batch, plane, x);
        if (is_train_and_not_global_stats) {
          const DType inp = input.get_ref(batch, plane, x);
          const AccReal proj = (inp - mean) * projScale;
          gradInput.get_ref(batch, plane, x) =
            ScalarConvert<AccReal, DType>::to((gradOut - proj - gradMean) * gradScale);
        } else {
          gradInput.get_ref(batch, plane, x) = ScalarConvert<AccReal, DType>::to(
            gradOut * gradScale);
        }
      }
    }
  }

  if (tensors.gradWeight.numElements() > 0 && threadIdx.x == 0 && (flags & WRITE_GAMMA_FLAG) != 0) {
    if ((flags & FIX_GAMMA_FLAG) == 0) {
      tensors.gradWeight[plane] = ScalarConvert<AccReal, DType>::to(dotP * invstd);
    } else {
      tensors.gradWeight[plane] = DType(0);
    }
  }

  if (tensors.gradBias.numElements() > 0 && threadIdx.x == 0 && (flags & WRITE_BETA_FLAG) != 0) {
    tensors.gradBias[plane] = ScalarConvert<AccReal, DType>::to(gradOutputSum);
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
    dim3 blocks(input.ChannelCount());
    dim3 threads(batchnorm::cuda::getNumThreads(input.InnerSize(), false));
    BatchNormalizationUpdateOutputInferenceKernel<DType, AccReal, DeviceTensor1,
      batchnorm::BNTensor3<DType>>
      <<< blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s) >>> (
      input, output, runningMean, runningVar, saveMean,
        saveInvStd, weight, bias, eps, flags);
  } else {
    dim3 blocks(input.ChannelCount());
    dim3 threads(batchnorm::cuda::getNumThreads(input.InnerSize(), false));
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

  CUDATensors<DeviceTensor1> tensors;

  tensors.gradWeight = devicetensor<AccReal, 1>(in_grad[batchnorm::kGamma]);
  tensors.gradBias = devicetensor<AccReal, 1>(in_grad[batchnorm::kBeta]);
  tensors.weight = devicetensor<AccReal, 1>(in_data[batchnorm::kGamma]);
  tensors.runningMean = devicetensor<AccReal, 1>(aux_states[batchnorm::kMovingMean]);
  tensors.runningVar = devicetensor<AccReal, 1>(aux_states[batchnorm::kMovingVar]);
  tensors.saveMean = devicetensor<AccReal, 1>(out_data[batchnorm::kMean]);
  tensors.saveInvStd = devicetensor<AccReal, 1>(out_data[batchnorm::kVar]);

  DCHECK_GT(tensors.weight.numElements(), 0);
#ifdef NDEBUG
  constexpr bool SMALLER_THREADS = false;
#else
  constexpr bool SMALLER_THREADS = true;
#endif
  dim3 blocks(gradOutput.ChannelCount());
  dim3 threads(batchnorm::cuda::getNumThreads(gradOutput.InnerSize(), SMALLER_THREADS));
  BatchNormalizationBackwardKernel<DType, AccReal, DeviceTensor1, batchnorm::BNTensor3<DType>>
    <<< blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s) >>> (
    input, gradOutput, gradInput, tensors, flags, momentum, eps);
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
  if (BatchNormOp<xpu, DType, AccReal>::IsWriting(req[batchnorm::kData])) {
    flags |= WRITE_DATA_FLAG;
  }
  if (BatchNormOp<xpu, DType, AccReal>::IsWriting(req[batchnorm::kGamma])) {
    flags |= WRITE_GAMMA_FLAG;
  }
  if (BatchNormOp<xpu, DType, AccReal>::IsWriting(req[batchnorm::kBeta])) {
    flags |= WRITE_BETA_FLAG;
  }
  return flags;
}

/*! \brief Forward batch-norm pass on GPU */
template<typename xpu, typename DType, typename AccReal>
void BatchNormOp<xpu, DType, AccReal>::DoForward(mshadow::Stream<gpu> *stream,
                                                 const OpContext &ctx,
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
void BatchNormOp<xpu, DType, AccReal>::DoBackward(mshadow::Stream<gpu> *stream,
                                                  const OpContext &ctx,
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

/*! \brief Create GPU operator for batch normalization */
template<>
Operator *CreateOp<gpu>(BatchNormParam param, const int dtype, const TShape& shape) {
  param.axis = mxnet::op::batchnorm::GetRealAxis(shape, param.axis);
  Operator *op = NULL;
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
  if (!param.use_global_stats && !param.cudnn_off && shape.ndim() <= 4
      && param.axis == mxnet::op::batchnorm::DEFAULT_AXIS) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new CuDNNBatchNormOp<DType>(param);
    })
  } else {
    MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DType, AccReal, {
      op = new BatchNormOp<gpu, DType, AccReal>(param);
    })
  }
#else
  MSHADOW_REAL_TYPE_SWITCH_EX(dtype,
                              DType,
                              AccReal,
                              { op = new BatchNormOp<gpu, DType, AccReal>(param); });
#endif
  return op;
}

}  // namespace op
}  // namespace mxnet
