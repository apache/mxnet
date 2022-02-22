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
 * \file count_sketch.cu
 * \brief count_sketch op
 * \author Chen Zhu, Yang Shi
 */
#include "./count_sketch-inl.h"
#include <mshadow/tensor.h>
#include <stdio.h>
#include <algorithm>

#define WARPS_PER_BLOCK   1
#define THREADS_PER_BLOCK 512

namespace mshadow {
namespace cuda {
// wrappers to deal with atomic add
// supporting only single precision
__device__ void atomic_add(float* dst, float val) {
  atomicAdd(dst, val);
}

// for double precision
__device__ void atomic_add(double* address, double val) {
  // code example in the official document at:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
  // #atomic-functions

  // NOLINT_NEXT_LINE(runtime/int)
  unsigned long long int* address_as_ull = (unsigned long long int*)address;  // NOLINT(*)
  unsigned long long int old             = *address_as_ull, assumed;          // NOLINT(*)
  do {
    assumed = old;
    old     = atomicCAS(
        address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN
    // (since NaN != NaN)
  } while (assumed != old);
}

template <typename DType>
__global__ void sketch_forward_kernel(const int nthreads,
                                      DType* out,
                                      const DType* h,
                                      const DType* s,
                                      const DType* in,
                                      const int n_smaples,
                                      const int in_dim,
                                      const int out_dim) {
  // input: n_smaples * in_dim
  // output: n_smaples * out_dim
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nthreads) {
    return;
  }
  // nthreads is the maximum of thread indices, should be equal to in_dim
  // index is point index
  const int i_indim  = index % in_dim;
  const int i_sample = index / in_dim;

  // get the target location in the output
  const int target = i_sample * out_dim + h[i_indim];
  atomic_add(out + target, s[i_indim] * in[index]);
}

template <typename DType>
__global__ void sketch_backward_kernel(const int nthreads,
                                       DType* in_grad,
                                       const DType* h,
                                       const DType* s,
                                       const DType* out_grad,
                                       const int n_smaples,
                                       const int in_dim,
                                       const int out_dim) {
  // only calculate gradient regarding x
  // can also calculate gradient regarding s if needed
  const int index    = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nthreads) {
    return;
  }
  const int i_indim  = index % in_dim;
  const int i_sample = index / in_dim;
  const int i_outdim = i_sample * out_dim + h[i_indim];
  in_grad[index]     = out_grad[i_outdim] * s[i_indim];
}

}  // namespace cuda

// CountSketch Forward
template <typename DType>
inline void CountSketchForward(const Tensor<gpu, 2, DType>& out,
                               const Tensor<gpu, 2, DType>& in,
                               const Tensor<gpu, 1, DType>& h,
                               const Tensor<gpu, 1, DType>& s,
                               const int n_samples,
                               const int processing_batch_size,
                               const int in_dim,
                               const int out_dim) {
  DType* out_ptr      = out.dptr_;
  const DType* in_ptr = in.dptr_;
  const DType* h_ptr  = h.dptr_;
  const DType* s_ptr  = s.dptr_;
  int upper_bound     = n_samples / processing_batch_size;
  if (n_samples % processing_batch_size == 0) {
    upper_bound = upper_bound - 1;
  }
  // guarantee there are at least one iteration
  upper_bound = upper_bound > 0 ? upper_bound : 0;
  int bstart  = 0;
  for (int i = 0; i <= upper_bound; i++) {
    const int batchlen = min(processing_batch_size, n_samples - bstart);
    const int nthreads = batchlen * in_dim;
    // to make number of threads the same as input
    const int threads_per_block = min(THREADS_PER_BLOCK, nthreads);
    int nblocks                 = (nthreads + threads_per_block - 1) / threads_per_block;
    cuda::sketch_forward_kernel<DType><<<nblocks, threads_per_block>>>(nthreads,
                                                                       out_ptr + bstart * out_dim,
                                                                       h_ptr,
                                                                       s_ptr,
                                                                       in_ptr + bstart * in_dim,
                                                                       batchlen,
                                                                       in_dim,
                                                                       out_dim);
    cudaError_t err = cudaDeviceSynchronize();
    CHECK_EQ(err, cudaSuccess) << "Error occured! CUDA: " << cudaGetErrorString(err);
    bstart = (i + 1) * batchlen;
  }
}

template <typename DType>
inline void CountSketchBackward(const Tensor<gpu, 2, DType>& in_grad,
                                const Tensor<gpu, 2, DType>& out_grad,
                                const Tensor<gpu, 1, DType>& h,
                                const Tensor<gpu, 1, DType>& s,
                                const int n_samples,
                                const int processing_batch_size,
                                const int in_dim,
                                const int out_dim) {
  DType* in_grad_ptr        = in_grad.dptr_;
  const DType* out_grad_ptr = out_grad.dptr_;
  const DType* h_ptr        = h.dptr_;
  const DType* s_ptr        = s.dptr_;
  int upper_bound           = n_samples / processing_batch_size;
  if (n_samples % processing_batch_size == 0) {
    upper_bound = upper_bound - 1;
  }
  // guarantee there are at least one iteration
  upper_bound = upper_bound > 0 ? upper_bound : 0;
  int bstart  = 0;
  for (int i = 0; i <= upper_bound; i++) {
    const int batchlen = min(processing_batch_size, n_samples - bstart);
    const int nthreads = batchlen * in_dim;
    // to make number of threads the same as input
    const int threads_per_block = min(THREADS_PER_BLOCK, nthreads);
    int nblocks                 = (nthreads + threads_per_block - 1) / threads_per_block;
    cuda::sketch_backward_kernel<DType>
        <<<nblocks, threads_per_block>>>(nthreads,
                                         in_grad_ptr + bstart * in_dim,
                                         h_ptr,
                                         s_ptr,
                                         out_grad_ptr + bstart * out_dim,
                                         batchlen,
                                         in_dim,
                                         out_dim);
    cudaError_t err = cudaDeviceSynchronize();
    CHECK_EQ(err, cudaSuccess) << "Error occured! CUDA: " << cudaGetErrorString(err);
    bstart = (i + 1) * batchlen;
  }
}
}  // namespace mshadow
namespace mxnet {
namespace op {
template <>
Operator* CreateOp<gpu>(CountSketchParam param, int dtype) {
  Operator* op = nullptr;
  switch (dtype) {
    case mshadow::kFloat32:
      op = new CountSketchOp<gpu, float>(param);
      break;
    case mshadow::kFloat64:
      op = new CountSketchOp<gpu, double>(param);
      break;
    case mshadow::kFloat16:
      LOG(FATAL) << "float16 count sketch layer is currently"
                    "not supported.";
      break;
    default:
      LOG(FATAL) << "Unsupported type " << dtype;
  }
  return op;
}
}  // namespace op
}  // namespace mxnet
