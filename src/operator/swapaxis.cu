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
Operator *CreateOp<gpu>(SwapAxisParam param) {
  return new SwapAxisOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

