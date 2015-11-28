/*!
 * Copyright (c) 2015 by Contributors
 * \file upsampling.cc
 * \brief
 * \author Bing Xu
*/


#include "./upsampling-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(UpSamplingParam param) {
  return new UpSamplingOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
