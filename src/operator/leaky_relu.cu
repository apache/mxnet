/*!
 * Copyright (c) 2015 by Contributors
 * \file leaky_relu.cc
 * \brief
 * \author Bing Xu
*/

#include "./leaky_relu-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(LeakyReLUParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op =  new LeakyReLUOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet

