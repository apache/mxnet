/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm.cu
 * \brief
 * \author Bing Xu
*/

#include "./batch_norm-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(BatchNormParam param) {
  return new BatchNormOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

