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
 * \file reduce.cuh
 * \brief helper functions to do reduction
 * \author Tianqi Chen
 */
#ifndef MSHADOW_CUDA_REDUCE_CUH_
#define MSHADOW_CUDA_REDUCE_CUH_

namespace mshadow {
namespace cuda {
/*
 * \brief reduce over the dimension x
 * \tparam Reducer reducer
 * \tparam x_bits dimension = 1<<x_bits
 * \tparam DType content data type
 */
template<typename Reducer, int x_bits, typename DType>
inline __device__ void Reduce1D(volatile DType buf[1 << x_bits]);
/*
 * \brief reduce over the dimension x
 * \tparam Reducer reducer
 * \tparam xmax_bits maximum size of buffer
 * \tparam DType content data type
 * \param xsize size of x dimension, not sure if aligned
 */
template<typename Reducer, int xmax_bits, typename DType>
inline __device__ void
Reduce1DNotAlign(volatile DType buf[1 << xmax_bits], int xsize);
// ===============================================x===
//  implementations afterwards,
//  no need to read if only use the functions
// --------------------------------------------------
#ifdef  __DEVICE_EMULATION__
#define __syncwarp() __syncthreads()
#else
#if CUDA_VERSION < 9000
#define __syncwarp()
#endif
#endif

template<typename Reducer, int x_bits, typename DType>
inline __device__ void ReduceX(volatile DType  buf[], int tid) {
  if (x_bits >= 10) {
    if (tid < 512) Reducer::Reduce(buf[tid] , buf[tid + 512]);
    __syncthreads();
  }
  if (x_bits >= 9) {
    if (tid < 256) Reducer::Reduce(buf[tid] , buf[tid + 256]);
    __syncthreads();
  }
  if (x_bits >= 8) {
    if (tid < 128) Reducer::Reduce(buf[tid] , buf[tid + 128]);
    __syncthreads();
  }
  if (x_bits >= 7) {
    if (tid < 64) Reducer::Reduce(buf[tid] , buf[tid + 64]);
    __syncthreads();
  }
  if (x_bits >= 6) {
    if (tid < 32) Reducer::Reduce(buf[tid] , buf[tid + 32]);
    __syncthreads();
  }
  // in warp optimization
  if (x_bits >= 5) {
    if (tid < 16) Reducer::Reduce(buf[tid] , buf[tid + 16]);
#if MSHADOW_OLD_CUDA
    __syncthreads();
#else
    __syncwarp();
#endif
  }
  if (x_bits >= 4) {
    if (tid < 8) Reducer::Reduce(buf[tid] , buf[tid + 8]);
    __syncwarp();
  }
  if (x_bits >= 3) {
    if (tid < 4) Reducer::Reduce(buf[tid] , buf[tid + 4]);
    __syncwarp();
  }
  if (x_bits >= 2) {
    if (tid < 2) Reducer::Reduce(buf[tid] , buf[tid + 2]);
    __syncwarp();
  }
  if (x_bits >= 1) {
    if (tid < 1) Reducer::Reduce(buf[tid] , buf[tid + 1]);
    __syncwarp();
  }
}
template<typename Reducer, int x_bits, typename DType>
inline __device__ void Reduce1D(volatile DType buf[1 << x_bits]) {
  ReduceX<Reducer, x_bits>(buf, threadIdx.x);
}
// reduce with a upper bound
#define __RD_NON_ALIGN(els, x_bits)                                     \
  els                                                                   \
  if (xmax_bits >= x_bits && x_size >= (1 << x_bits)) {                 \
    if (tid < (1 << x_bits) && tid + (1 << x_bits) < x_size) {          \
      Reducer::Reduce(buf[tid] , buf[tid + (1 << x_bits)]);             \
    }                                                                   \
    __syncthreads();                                                    \
    ReduceX<Reducer, x_bits>(buf, tid);                                 \
  }                                                                     \

template<typename Reducer, int xmax_bits, typename DType>
inline __device__ void Reduce1DNotAlign(volatile DType buf[], int x_size) {
  int tid = threadIdx.x;
  __RD_NON_ALIGN(, 8)
  __RD_NON_ALIGN(else, 7)
  __RD_NON_ALIGN(else, 6)
  __RD_NON_ALIGN(else, 5)
  __RD_NON_ALIGN(else, 4)
  __RD_NON_ALIGN(else, 3)
  __RD_NON_ALIGN(else, 2)
  __RD_NON_ALIGN(else, 1)
}
}  // namespace cuda
}  // namespace mshadow
#endif  // MSHADOW_CUDA_REDUCE_CUH_

