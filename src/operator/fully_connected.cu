/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connected.cu
 * \brief fully connect operator
*/
#include "./fully_connected-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(FullyConnectedParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new FullyConnectedOp<gpu, DType>(param);
  })
  return op;
}
}  // namespace op
}  // namespace mxnet
