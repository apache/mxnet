/*!
 * Copyright (c) 2017 by Contributors
 * \file grid_generator.cu
 * \brief
 * \author Xu Dong
*/

#include "./grid_generator-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(GridGeneratorParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new GridGeneratorOp<gpu, DType>(param);
  })
  return op;
}
}  // namespace op
}  // namespace mxnet
