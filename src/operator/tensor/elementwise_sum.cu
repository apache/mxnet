/*!
 * Copyright (c) 2015 by Contributors
 * \file elementwise_sum.cu
 * \brief elementwise sum operator
*/
#include "./elementwise_sum-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ElementWiseSumParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ElementWiseSumOp<gpu, DType>(param);
  });
  return op;
}
}  // namespace op
}  // namespace mxnet
