/*!
 * Copyright (c) 2016 by Contributors
 * \file q_fully_connected.cu
 * \brief Quantized FC operator
 * \author HPI-DeepLearning
*/
#include "./q_fully_connected-inl.h"
#include <mshadow/tensor.h>

namespace mshadow {

  inline void QFullyConnectedForward(int m, int n, int k,
                                     const Tensor<gpu, 2, float> &data,
                                     Tensor<gpu, 1, float> &workspace,
                                     mxnet::op::xnor_cpu::BINARY_WORD* wmat_binarized,
                                     Tensor<gpu, 2, float> &out) {
    CHECK(false) << "cuda with pre-binarized weights not implemented";
  }

  inline void QFullyConnectedForward(int m, int n, int k,
                                     const Tensor<gpu, 2, float> &data,
                                     Tensor<gpu, 1, float> &workspace,
                                     const Tensor<gpu, 2, float> &wmat,
                                     Tensor<gpu, 2, float> &out) {
    // !deprecated! will be removed later
    //cuda::QFullyConnectedForward(data, wmat, out);
  }

  template<typename DType>
  inline void QFullyConnectedForward(int m, int n, int k,
                                     const Tensor<gpu, 2, DType> &data,
                                     Tensor<gpu, 1, DType> &workspace,
                                     mxnet::op::xnor_cpu::BINARY_WORD* wmat_binarized,
                                     Tensor<gpu, 2, DType> &out) {
    CHECK(false) << "only float supported";
  }

  template<typename DType>
  inline void QFullyConnectedForward(int m, int n, int k,
                                     const Tensor<gpu, 2, DType> &data,
                                     Tensor<gpu, 1, DType> &workspace,
                                     const Tensor<gpu, 2, DType> &wmat,
                                     Tensor<gpu, 2, DType> &out) {
    CHECK(false) << "only float supported";
  }
} // namespace mshadow


namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(QFullyConnectedParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new QFullyConnectedOp<gpu, DType>(param);
  })
  return op;
}
}  // namespace op
}  // namespace mxnet
