/*!
 * Copyright (c) 2017 by Contributors
 * \file kernel_launch_op.h
 * \brief
*/
#ifndef MXNET_OPERATOR_KERNEL_LAUNCH_OP_H_
#define MXNET_OPERATOR_KERNEL_LAUNCH_OP_H_

#include <mxnet/base.h>
#include "special_functions-inl.h"

namespace mxnet {
namespace op {
namespace kernel_launch_op {

/*! \brief sigmoid unit */
struct sigmoid {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *xs) {
    out[i] = DType(DType(1.0f) / (DType(1.0f) + expf(-xs[i])));
  }
};
struct sigmoid_grad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *xs) {
    DType x = xs[i];
    out[i] = DType(x * (DType(1.0f) - x));
  }
};
/*! \brief Rectified Linear Operation */
struct relu {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *xs) {
    DType x = xs[i];
    out[i] = DType(x > DType(0.0f) ? x : DType(0.0f));
  }
};
struct relu_grad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *xs) {
    DType x = xs[i];
    out[i] = DType(x > DType(0.0f) ? DType(1.0f) : DType(0.0f));
  }
};

}  // namespace kernel_launch_op
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_KERNEL_LAUNCH_OP_H_
