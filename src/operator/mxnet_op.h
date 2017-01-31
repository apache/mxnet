/*!
 * Copyright (c) 2017 by Contributors
 * \file mxnet_op.h
 * \brief
 * \author Junyuan Xie
*/
#ifndef MXNET_OPERATOR_MXNET_OP_H_
#define MXNET_OPERATOR_MXNET_OP_H_

#include <mxnet/base.h>
#include <algorithm>

namespace mxnet {
namespace op {
namespace mxnet_op {
#ifdef __CUDA_ARCH__
__constant__ const float PI = 3.14159265358979323846;
#else
const float PI = 3.14159265358979323846;
using std::isnan;
#endif


template<typename OP, typename xpu>
struct Kernel;

template<typename OP>
struct Kernel<OP, cpu> {
  template<typename ...Args>
  inline static void Launch(mshadow::Stream<cpu> *s, int N, Args... args) {
#if (MXNET_USE_CUDA == 0)
    #pragma omp parallel for
#endif
    for (int i = 0; i < N; ++i) {
      OP::Map(i, args...);
    }
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

struct clip {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* datas,
                                  DType a_min, DType a_max) {
    DType data = datas[i];
    if (data > a_max) {
      out[i] = a_max;
    } else if (data < a_min) {
      out[i] = a_min;
    } else {
      out[i] = data;
    }
  }
};

struct clip_grad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* grad, const DType* datas,
                                  DType a_min, DType a_max) {
    DType data = datas[i];
    if (data > a_max) {
      out[i] = 0;
    } else if (data < a_min) {
      out[i] = 0;
    } else {
      out[i] = grad[i];
    }
  }
};

/*!
 * \brief Primary template definition for operator repeat.
 * The function Map is a dummy here.
 * Will be specialized later for different cases.
 * The function Map is under looping through every
 * element of the input data.
 */
template<size_t nDims, int axis>
struct repeat {
  /*!
   * \brief
   * \param i index of the input array in 1D form
   * \param out output array
   * \param datas input array
   * \param repeats number of repeating times
   * \param n1 size of the input array in the first dim
   * \param n2 size of the input array in the second dim (dummy for arrays of dims > 1)
   * \param n3 size of the input array in the third dim (dummy for arrays of dims > 2)
   */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* datas, int repeats,
                                  size_t n1 = 0, size_t n2 = 0, size_t n3 = 0) {}  
};

/*!
 * \brief Specialized template for operator repeat.
 * This implementation handles the most common case
 * when nDims = 3 and axis = 1.
 */
template<>
struct repeat<3, 1> {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* datas, int repeats,
                                  size_t n1 = 0, size_t n2 = 0, size_t n3 = 0) {
    DType data = datas[i];
    size_t n23 = n2 * n3;
    // calculate the 3D index of the input 3D array from its 1D index i
    int i1 = i / n23;
    int i2 = (i % n23) / n3;
    int i3 = i - i1 * n23 - i2 * n3;

    // calculate the corresponding 3D index of the output 3D array
    // j1 = i1, j3 = i3, so the assignment will be skipped.
    int j2 = i2 * repeats;
    int n23_b = n23 * repeats;
    for (int k = 0; k < repeats; ++k) {
      out[i1*n23_b+(j2+k)*n3+i3] = data;
    }
  }  
};

/*!
 * \brief Specialized template for operator repeat.
 * This implementation handles the use case where axis
 * is not specified by users or the input array
 * is a 1D array. If the input array is a multi-dim array,
 * it will be flattened into a 1D array before being passed to
 * the kernel. Hence, nDims = 1 in this case.
 */
template<>
struct repeat<1, 1> {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* datas, int repeats,
                                  size_t n1 = 0, size_t n2 = 0, size_t n3 = 0) {
    DType data = datas[i];
    int j = i * repeats; // starting index of every element in out
    for (int k = 0; k < repeats; ++k) {
      out[j+k] = data;
    }
  }
};

/*!
 * \brief Specialize template for operator repeat.
 * This implementation handles the use case where
 * nDims = 2 and axis = 0.
 */
template<>
struct repeat<2, 0> {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* datas, int repeats,
                                  size_t n1 = 0, size_t n2 = 0, size_t n3 = 0) {
    DType data = datas[i];
    int i1 = i / n2;
    int i2 = i - i1 * n2;
    int j1 = i1 * repeats; // starting index of every element of out in the first dim
    for (int k = 0; k < repeats; ++k) {
      out[(j1+k)*n2+i2] = data;
    }
  }
};

/*!
 * \brief Specialize template for operator repeat.
 * This implementation handles the use case where
 * nDims = 2 and axis = 1.
 */
template<>
struct repeat<2, 1> {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* datas, int repeats,
                                  size_t n1 = 0, size_t n2 = 0, size_t n3 = 0) {
    DType data = datas[i];
    int j = i * repeats;
    for (int k = 0; k < repeats; ++k) {
      out[j+k] = data;
    }
  }
};

}  // namespace mxnet_op
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MXNET_OP_H_
