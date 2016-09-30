/*!
 * Copyright (c) 2015 by Contributors
 * \file scale.cu
 * \brief scale operator
*/
#include "./scale-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ScaleParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ScaleOp<gpu, DType>(param);
  });
  return op;
}
}  // namespace op
}  // namespace mxnet
