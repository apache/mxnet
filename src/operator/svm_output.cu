/*!
 * Copyright (c) 2015 by Contributors
 * \file svm_output.cu
 * \brief
 * \author Jonas Amaro
*/

#include "./svm_output-inl.h"

namespace mshadow {
  template<typename DType>
  inline void L1_SVM(const DType & margin,
                   const DType & reg_coef,
                   Tensor<gpu, 2, DType> dst,
                   const Tensor<gpu, 1, DType> & label,
                   const Tensor<gpu, 2, DType> & src) {
    LOG(FATAL) << "Not Implemented.";
  }
  template<typename DType>
  inline void L2_SVM(const DType & margin,
               const DType & reg_coef,
               Tensor<gpu, 2, DType> dst,
               const Tensor<gpu, 1, DType> & label,
               const Tensor<gpu, 2, DType> & src) {
    LOG(FATAL) << "Not Implemented.";
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

