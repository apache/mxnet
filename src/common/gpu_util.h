/*!
 * Copyright (c) 2016 by Contributors
 * \file cuda_utils.h
 * \brief CUDA debugging utilities.
 */

#ifndef MXNET_UTIL_GPU_UTIL_H_
#define MXNET_UTIL_GPU_UTIL_H_


#if MXNET_USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>



template <typename Dtype>
inline __device__ Dtype mxnet_gpu_atomic_add(const Dtype val, Dtype* address);

template <>
inline __device__
float mxnet_gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

// double atomicAdd implementation taken from:
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz3PVCpVsEG
template <>
inline __device__
double mxnet_gpu_atomic_add(const double val, double* address) {
  unsigned long long int* address_as_ull =  // NOLINT(runtime/int)
      // NOLINT_NEXT_LINE(runtime/int)
      reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;  // NOLINT(runtime/int)
  unsigned long long int assumed;  // NOLINT(runtime/int)
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}




#endif  // mxnet_UTIL_GPU_UTIL_H_
#endif  // MXNET_USE_CUDNN
