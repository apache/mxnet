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

}  // namespace mxnet_op
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MXNET_OP_H_
