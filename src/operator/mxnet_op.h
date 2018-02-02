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
#include "./operator_tune.h"
#include "../engine/openmp.h"

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
  return engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
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


#define MXNET_NDIM_SWITCH(NDim, ndim, ...)         \
  if (NDim == 0) {                                 \
  } else if (NDim == 1) {                          \
    const int ndim = 1;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 2) {                          \
    const int ndim = 2;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 3) {                          \
    const int ndim = 3;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 4) {                          \
    const int ndim = 4;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 5) {                          \
    const int ndim = 5;                            \
    {__VA_ARGS__}                                  \
  } else {                                         \
    LOG(FATAL) << "ndim=" << NDim << "too large "; \
  }

#define MXNET_NO_INT8_TYPE_SWITCH(type, DType, ...)        \
  switch (type) {                                          \
  case mshadow::kFloat32:                                  \
    {                                                      \
      typedef float DType;                                 \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat64:                                  \
    {                                                      \
      typedef double DType;                                \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat16:                                  \
    {                                                      \
      typedef mshadow::half::half_t DType;                 \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kUint8:                                    \
    LOG(FATAL) << "This operation does not "               \
                  "support int8 or uint8";                 \
    break;                                                 \
  case mshadow::kInt8:                                     \
    LOG(FATAL) << "This operation does not "               \
                  "support int8 or uint8";                 \
    break;                                                 \
  case mshadow::kInt32:                                    \
    {                                                      \
      typedef int32_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt64:                                    \
    {                                                      \
      typedef int64_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  default:                                                 \
    LOG(FATAL) << "Unknown type enum " << type;            \
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
  for (int i = 0; i < ndim; ++i) {
    ret += coord[i] * stride[i];
  }
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

/* Increment coordinates and modify index */
template<int ndim>
MSHADOW_XINLINE void inc(Shape<ndim>* coord, const Shape<ndim>& shape,
                         index_t* idx, const Shape<ndim>& stride) {
  ++(*coord)[ndim-1];
  *idx += stride[ndim-1];
  #pragma unroll
  for (int i = ndim - 1; i > 0 && (*coord)[i] >= shape[i]; --i) {
    (*coord)[i] -= shape[i];
    ++(*coord)[i-1];
    *idx = *idx + stride[i-1] - shape[i] * stride[i];
  }
}

/* Increment coordinates and modify index */
template<int ndim>
MSHADOW_XINLINE void inc(Shape<ndim>* coord, const Shape<ndim>& shape,
                         index_t* idx1, const Shape<ndim>& stride1,
                         index_t* idx2, const Shape<ndim>& stride2) {
  ++(*coord)[ndim-1];
  *idx1 += stride1[ndim-1];
  *idx2 += stride2[ndim-1];
  #pragma unroll
  for (int i = ndim - 1; i > 0 && (*coord)[i] >= shape[i]; --i) {
    (*coord)[i] -= shape[i];
    ++(*coord)[i-1];
    *idx1 = *idx1 + stride1[i-1] - shape[i] * stride1[i];
    *idx2 = *idx2 + stride2[i-1] - shape[i] * stride2[i];
  }
}

/*!
 * \brief Simple copy data from one blob to another
 * \param to Destination blob
 * \param from Source blob
 */
template <typename xpu>
MSHADOW_CINLINE void copy(mshadow::Stream<xpu> *s, const TBlob& to, const TBlob& from) {
  CHECK_EQ(from.Size(), to.Size());
  CHECK_EQ(from.dev_mask(), to.dev_mask());
  MSHADOW_TYPE_SWITCH(to.type_flag_, DType, {
    if (to.type_flag_ == from.type_flag_) {
      mshadow::Copy(to.FlatTo1D<xpu, DType>(), from.FlatTo1D<xpu, DType>(), s);
    } else {
      MSHADOW_TYPE_SWITCH(from.type_flag_, SrcDType, {
        to.FlatTo1D<xpu, DType>(s) = mshadow::expr::tcast<DType>(from.FlatTo1D<xpu, SrcDType>(s));
      })
    }
  })
}

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

/*! \brief Binary op backward gradient OP wrapper (tuned) */
template<typename GRAD_OP>
struct backward_grad_tuned : public backward_grad<GRAD_OP>, public tunable {
  using backward_grad<GRAD_OP>::Map;
};

/*! \brief Select assignment operation based upon the req value
 * Also useful for mapping mshadow Compute (F<OP>) to Kernel<OP>::Launch
 */
template<typename OP, int req>
struct op_with_req {
  typedef OP Operation;

  /*! \brief input is one tensor */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *in) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i]));
  }

  /*! \brief inputs are two tensors */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *lhs, const DType *rhs) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], rhs[i]));
  }

  /*! \brief input is tensor and a scalar value */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *in, const DType value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i], value));
  }

  /*! \brief input is tensor and two scalar value */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *in,
                                  const DType value_1, const DType value_2) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i], value_1, value_2));
  }

  /*! \brief No inputs (ie fill to constant value) */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out) {
    KERNEL_ASSIGN(out[i], req, OP::Map());
  }

  /*! \brief input is single scalar value */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(value));
  }

  /*! \brief inputs are two tensors and a scalar value */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out,
                                  const DType *input_1, const DType *input_2, const DType value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(input_1[i], input_2[i], value));
  }

  /*! \brief inputs are three tensors (ie backward grad with binary grad function) */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out,
                                  const DType *input_1,
                                  const DType *input_2,
                                  const DType *input_3) {
    KERNEL_ASSIGN(out[i], req, OP::Map(input_1[i], input_2[i], input_3[i]));
  }
};

template<typename OP, typename xpu>
struct Kernel;

/*!
 * \brief CPU Kernel launcher
 * \tparam OP Operator to launch
 */
template<typename OP>
struct Kernel<OP, cpu> {
  /*!
   * \brief Launch a generic CPU kernel.
   * When using this for a new kernel op, add declaration and tuning objects to
   * operator_tune.cc
   * \tparam Args Varargs type to eventually pass to the OP::Map() functoion
   * \param N Number of iterations
   * \param args Varargs to eventually pass to the OP::Map() functoion
   */
  template<typename ...Args>
  inline static bool Launch(mshadow::Stream<cpu> *, const int N, Args... args) {
#ifdef _OPENMP
    const int omp_threads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
    if (omp_threads < 2) {
      for (int i = 0; i < N; ++i) {
        OP::Map(i, args...);
      }
    } else {
      #pragma omp parallel for num_threads(omp_threads)
      for (int i = 0; i < N; ++i) {
        OP::Map(i, args...);
      }
    }
#else
    for (int i = 0; i < N; ++i) {
      OP::Map(i, args...);
    }
#endif
    return true;
  }

  /*!
   * \brief Launch CPU kernel which has OMP tuning data available.
   * When using this for a new kernel op, add declaration and tuning objects to
   * operator_tune.cc
   * \tparam PRIMITIVE_OP The primitive operation to use for tuning
   * \tparam DType Data type
   * \tparam Args Varargs type to eventually pass to the OP::Map() functoion
   * \param N Number of iterations
   * \param dest Destination pointer (used to infer DType)
   * \param args Varargs to eventually pass to the OP::Map() functoion
   */
  template<typename PRIMITIVE_OP, typename DType, typename ...Args>
  static void LaunchTuned(mshadow::Stream<cpu> *, const int N, Args... args) {
#ifdef _OPENMP
    const int omp_threads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
    if (omp_threads < 2 || !tuned_op<PRIMITIVE_OP, DType>::UseOMP(
      static_cast<size_t>(N), static_cast<size_t>(omp_threads))) {
      for (int i = 0; i < N; ++i) {
        OP::Map(i, args...);
      }
    } else {
      #pragma omp parallel for num_threads(omp_threads)
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

  /*!
   * \brief Launch custom-tuned kernel where each thread is set to
   *        operate on a contiguous partition
   * \tparam Args Varargs type to eventually pass to the OP::Map() functoion
   * \param N Number of iterations
   * \param args Varargs to eventually pass to the UseOMP() and OP::Map() functions
   */
  template<typename ...Args>
  inline static void LaunchEx(mshadow::Stream<cpu> *s, const int N, Args... args) {
#ifdef _OPENMP
    const int omp_threads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
    if (omp_threads < 2) {
      OP::Map(0, N, args...);
    } else {
      const int length = (N + omp_threads - 1) / omp_threads;
      #pragma omp parallel for num_threads(omp_threads)
      for (int i = 0; i < N; i += length) {
        OP::Map(i, i + length > N ? N - i : length, args...);
      }
    }
#else
    OP::Map(0, N, args...);
#endif
  }

  /*!
   * \brief Launch a tunable OP with implicitly-supplied data type
   * \tparam DType Data type
   * \tparam T OP type
   * \tparam Args Varargs type to eventually pass to the OP::Map() functoion
   * \param s Stream (usually null for CPU)
   * \param N Number of iterations
   * \param args Varargs to eventually pass to the OP::Map() functoion
   * \return Always true
   */
  template<typename DType, typename T = OP, typename ...Args>
  static MSHADOW_CINLINE
  typename std::enable_if<std::is_base_of<tunable, T>::value, bool>::type
  Launch(mshadow::Stream<cpu> *s, const int N, DType *dest, Args... args) {
    LaunchTuned<T, DType>(s, N, dest, args...);
    return true;
  }

  /*!
   * \brief Launch a tunable OP wrapper with explicitly-supplied data type (ie op_with_req)
   * \tparam DType Data type
   * \tparam T Wrapper type
   * \tparam Args Varargs type to eventually pass to the OP::Map() functoion
   * \param s Stream (usually null for CPU)
   * \param N Number of iterations
   * \param args Varargs to eventually pass to the OP::Map() functoion
   * \return Always true
   */
  template<typename DType, typename T = OP, typename ...Args>
  static MSHADOW_CINLINE
  typename std::enable_if<std::is_base_of<tunable, typename T::Operation>::value, bool>::type
  Launch(mshadow::Stream<cpu> *s, const int N, DType *dest, Args... args) {
    LaunchTuned<typename T::Operation, DType>(s, N, dest, args...);
    return true;
  }
};



#ifdef __CUDACC__
template<typename OP, typename ...Args>
__global__ void mxnet_generic_kernel(int N, Args... args) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    OP::Map(i, args...);
  }
}

template<typename OP, typename ...Args>
__global__ void mxnet_generic_kernel_ex(int N, Args... args) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    OP::Map(i, 1, args...);
  }
}

template<typename OP>
struct Kernel<OP, gpu> {
  /*! \brief Launch GPU kernel */
  template<typename ...Args>
  inline static void Launch(mshadow::Stream<gpu> *s, int N, Args... args) {
    using namespace mshadow::cuda;
    int ngrid = std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
    mxnet_generic_kernel<OP, Args...>
      <<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
        N, args...);
    MSHADOW_CUDA_POST_KERNEL_CHECK(mxnet_generic_kernel);
  }

  template<typename ...Args>
  inline static void LaunchEx(mshadow::Stream<gpu> *s, const int N, Args... args) {
    using namespace mshadow::cuda;
    int ngrid = std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
    mxnet_generic_kernel_ex<OP, Args...>
      <<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
        N, args...);
    MSHADOW_CUDA_POST_KERNEL_CHECK(mxnet_generic_kernel_ex);
  }
};
#endif  // __CUDACC__

/*!
 * \brief Set to immediate scalar value kernel
 * \tparam val Scalar immediate
 */
template<int val>
struct set_to_int : public tunable {
  // mxnet_op version (when used directly with Kernel<>::Launch()) */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out) {
    out[i] = DType(val);
  }
  // mshadow_op version (when used with op_with_req<>)
  MSHADOW_XINLINE static int Map() {
    return val;
  }
};

/*!
 * \brief Special-case kernel shortcut for setting to zero and one
 */
using set_zero = set_to_int<0>;
using set_one  = set_to_int<1>;
}  // namespace mxnet_op

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MXNET_OP_H_
