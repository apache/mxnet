/*!
 * Copyright (c) 2015 by Contributors
 * \file swapaxis.cu
 * \brief
 * \author Ming Zhang
*/

#include "./swapaxis-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(SwapAxisParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op =  new SwapAxisOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet

