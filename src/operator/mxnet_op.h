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
 * \file mxnet_op.h
 * \brief
 * \author Junyuan Xie
*/
#ifndef MXNET_OPERATOR_MXNET_OP_H_
#define MXNET_OPERATOR_MXNET_OP_H_

#include <dmlc/omp.h>
#include <mxnet/base.h>
#include <mxnet/engine.h>
#include <mxnet/op_attr_types.h>
#include <algorithm>

#ifdef __CUDACC__
#include "../common/cuda_utils.h"
#endif  // __CUDACC__

namespace mxnet {
namespace op {
namespace mxnet_op {
using namespace mshadow;

#ifdef __CUDA_ARCH__
__constant__ const float PI = 3.14159265358979323846;
#else
const float PI = 3.14159265358979323846;
using std::isnan;
#endif

template<typename xpu>
int get_num_threads(const int N);

#ifdef __CUDACC__
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

inline cudaDeviceProp cuda_get_device_prop() {
  int device;
  CUDA_CALL(cudaGetDevice(&device));
  cudaDeviceProp deviceProp;
  CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));
  return deviceProp;
}

/*!
 * \brief Get the number of blocks for cuda kernel given N
 */
inline int cuda_get_num_blocks(const int N) {
  using namespace mshadow::cuda;
  return std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
}

template<>
inline int get_num_threads<gpu>(const int N) {
  using namespace mshadow::cuda;
  return kBaseThreadNum * cuda_get_num_blocks(N);
}
#endif  // __CUDACC__

template<>
inline int get_num_threads<cpu>(const int N) {
  return omp_get_max_threads();
}

/*! \brief operator request type switch */
#define MXNET_ASSIGN_REQ_SWITCH(req, ReqType, ...)  \
  switch (req) {                                    \
  case kNullOp:                                     \
    break;                                          \
  case kWriteInplace:                               \
  case kWriteTo:                                    \
    {                                               \
      const OpReqType ReqType = kWriteTo;           \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case kAddTo:                                      \
    {                                               \
      const OpReqType ReqType = kAddTo;             \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    break;                                          \
  }


/*!
 * \brief assign the val to out according
 * to request in Kernel::Launch
 * \param out the data to be assigned
 * \param req the assignment request
 * \param val the value to be assigned to out
 * \tparam OType output type
 * \tparam VType value type
 */
#define KERNEL_ASSIGN(out, req, val)  \
  {                                   \
    switch (req) {                    \
      case kNullOp:                   \
        break;                        \
      case kWriteTo:                  \
      case kWriteInplace:             \
        (out) = (val);                \
        break;                        \
      case kAddTo:                    \
        (out) += (val);               \
        break;                        \
      default:                        \
        break;                        \
    }                                 \
  }


/* \brief Compute flattened index given coordinates and shape. */
template<int ndim>
MSHADOW_XINLINE int ravel(const Shape<ndim>& coord, const Shape<ndim>& shape) {
  int ret = 0;
  #pragma unroll
  for (int i = 0; i < ndim; ++i) {
    ret = ret * shape[i] + (shape[i] > coord[i]) * coord[i];
  }
  return ret;
}


/* Compute coordinates from flattened index given shape */
template<int ndim>
MSHADOW_XINLINE Shape<ndim> unravel(const int idx, const Shape<ndim>& shape) {
  Shape<ndim> ret;
  #pragma unroll
  for (int i = ndim-1, j = idx; i >=0; --i) {
    int tmp = j / shape[i];
    ret[i] = j - tmp*shape[i];
    j = tmp;
  }
  return ret;
}


/* Compute dot product of two vector */
template<int ndim>
MSHADOW_XINLINE int dot(const Shape<ndim>& coord, const Shape<ndim>& stride) {
  int ret = 0;
  #pragma unroll
  for (int i = 0; i < ndim; ++i)
    ret += coord[i] * stride[i];
  return ret;
}


/* Combining unravel and dot */
template<int ndim>
MSHADOW_XINLINE int unravel_dot(const int idx, const Shape<ndim>& shape,
  const Shape<ndim>& stride) {
  int ret = 0;
  #pragma unroll
  for (int i = ndim-1, j = idx; i >=0; --i) {
    int tmp = j / shape[i];
    ret += (j - tmp*shape[i])*stride[i];
    j = tmp;
  }
  return ret;
}


/* Calculate stride of each dim from shape */
template<int ndim>
MSHADOW_XINLINE Shape<ndim> calc_stride(const Shape<ndim>& shape) {
  Shape<ndim> stride;
  index_t cumprod = 1;
  #pragma unroll
  for (int i = ndim - 1; i >= 0; --i) {
    stride[i] = (shape[i] > 1) ? cumprod : 0;
    cumprod *= shape[i];
  }
  return stride;
}


struct fill {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType val) {
    out[i] = val;
  }
};


struct set_zero {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out) {
    out[i] = static_cast<DType>(0);
  }
};

/*! \brief Binary op backward gradient OP wrapper */
template<typename GRAD_OP>
struct backward_grad {
  /* \brief Backward calc with grad
   * \param a - output grad
   * \param args... - data to grad calculation op (what this is -- input, output, etc. -- varies)
   * \return input grad
   */
  template<typename DType, typename ...Args>
  MSHADOW_XINLINE static DType Map(DType a, Args... args) {
    return DType(a * GRAD_OP::Map(args...));
  }
};

/*! \brief Select assignment operation based upon the req value
 * Also useful for mapping mshadow Compute (F<OP>) to Kernel<OP>::Launch
 */
template<typename OP, int req>
struct op_with_req {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *in) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i]));
  }

  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *lhs, const DType *rhs) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], rhs[i]));
  }

  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *in, const DType value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i], value));
  }
};

template<typename OP, typename xpu>
struct Kernel;


template<typename OP>
struct Kernel<OP, cpu> {
  template<typename ...Args>
  inline static void Launch(mshadow::Stream<cpu> *s, const int N, Args... args) {
#ifdef _OPENMP
    const int omp_cores = Engine::Get()->num_omp_threads_per_worker();
    if (omp_cores <= 1) {
      // Zero means not to use OMP, but don't interfere with external OMP behavior
      for (int i = 0; i < N; ++i) {
        OP::Map(i, args...);
      }
    } else {
      #pragma omp parallel for num_threads(omp_cores)
      for (int i = 0; i < N; ++i) {
        OP::Map(i, args...);
      }
    }
#else
    for (int i = 0; i < N; ++i) {
        OP::Map(i, args...);
    }
#endif
  }
};


#ifdef __CUDACC__
template<typename OP, typename ...Args>
__global__ void mxnet_generic_kernel(int N, Args... args) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    OP::Map(i, args...);
  }
}

template<typename OP>
struct Kernel<OP, gpu> {
  template<typename ...Args>
  inline static void Launch(mshadow::Stream<gpu> *s, int N, Args... args) {
    using namespace mshadow::cuda;
    int ngrid = std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
    mxnet_generic_kernel<OP, Args...>
      <<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
        N, args...);
  }
};
#endif  // __CUDACC__

}  // namespace mxnet_op
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MXNET_OP_H_
