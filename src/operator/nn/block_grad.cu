/*!
 * Copyright (c) 2015 by Contributors
 * \file block_grad.cc
 * \brief
 * \author Bing Xu
*/
#include "./block_grad-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new BlockGradientOp<gpu, DType>();
  });
  return op;
}

}  // namespace op
}  // namespace mxnet

