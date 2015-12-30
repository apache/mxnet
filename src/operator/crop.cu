/*!
 * Copyright (c) 2015 by Contributors
 * \file concat.cu
 * \brief
 * \author Wei Wu
*/

#include "./crop-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(CropParam param) {
  return new CropOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
