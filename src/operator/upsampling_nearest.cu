/*!
 * Copyright (c) 2015 by Contributors
 * \file upsampling.cc
 * \brief
 * \author Bing Xu
*/


#include "./upsampling_nearest-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(UpSamplingNearestParam param) {
  return new UpSamplingNearestOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
