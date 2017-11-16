/*!
 * Copyright (c) 2015 by Contributors
 * \file svm_output.cu
 * \brief
 * \author Jonas Amaro
*/

#include "./svm_output-inl.h"
#include <device_launch_parameters.h>
#include "mshadow/tensor.h"


namespace mshadow {

template<int n_bits, typename DType>
__global__  void L1_SVMKernel(const DType margin,
                              const DType reg_coef,
                              Tensor<gpu, 2, DType> dst,
                              const Tensor<gpu, 1, DType> label,
                              const Tensor<gpu, 2, DType> src) {
  const index_t nmax = dst.size(1);
  const unsigned n_size = 1 << n_bits;
  const int y = blockIdx.x;
  const int n = threadIdx.x;
  const index_t k = static_cast<int>(label[y]);
  for (index_t n_index = n; n_index < nmax; n_index += n_size) {
    if (n_index == k) {
      dst[y][k] = -DType(margin > src[y][k]) * reg_coef;
    } else {
      dst[y][n_index] = DType(margin > -src[y][n_index]) * reg_coef;
    }
  }
}

template<typename DType>
inline void L1_SVM(const DType & margin,
                   const DType & reg_coef,
                   Tensor<gpu, 2, DType> dst,
                   const Tensor<gpu, 1, DType> & label,
                   const Tensor<gpu, 2, DType> & src) {
  dim3 dimBlock(cuda::kBaseThreadNum);
  dim3 dimGrid(dst.size(0));
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  L1_SVMKernel<cuda::kBaseThreadBits, DType> <<<dimGrid, dimBlock, 0, stream >>>
    (margin, reg_coef, dst, label, src);
}


template<int n_bits, typename DType>
__global__  void L2_SVMKernel(const DType margin,
                              const DType reg_coef,
                              Tensor<gpu, 2, DType> dst,
                              const Tensor<gpu, 1, DType> label,
                              const Tensor<gpu, 2, DType> src) {
  const index_t nmax = dst.size(1);
  const unsigned n_size = 1 << n_bits;
  const int y = blockIdx.x;
  const int n = threadIdx.x;
  const index_t k = static_cast<int>(label[y]);
  for (index_t n_index = n; n_index < nmax; n_index += n_size) {
    if (n_index == k) {
      dst[y][k] = margin > src[y][k] ? 2 * (margin - src[y][k]) : DType(0.0f);
      dst[y][k] *= -reg_coef;
    } else {
      dst[y][n_index] = margin > -src[y][n_index] ? (-2)*(margin + src[y][n_index]) : DType(0.0f);
      dst[y][n_index] *= -reg_coef;
    }
  }
}

template<typename DType>
inline void L2_SVM(const DType & margin,
                   const DType & reg_coef,
                   Tensor<gpu, 2, DType> dst,
                   const Tensor<gpu, 1, DType> & label,
                   const Tensor<gpu, 2, DType> & src) {
  dim3 dimBlock(cuda::kBaseThreadNum);
  dim3 dimGrid(dst.size(0));
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  L2_SVMKernel<cuda::kBaseThreadBits, DType> <<<dimGrid, dimBlock, 0, stream >>>
    (margin, reg_coef, dst, label, src);
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(SVMOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SVMOutputOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

