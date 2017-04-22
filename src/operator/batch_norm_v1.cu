/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm_v1.cu
 * \brief
 * \author Bing Xu
*/

#include "batch_norm_v1-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(BatchNormV1Param param, int dtype) {
  return new BatchNormV1Op<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

