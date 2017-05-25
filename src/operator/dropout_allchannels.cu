/*!
 * Copyright (c) 2015, 2016 by Contributors
 * \file dropout_allchannels.cc
 * \brief
 * \author Bing Xu, Kai Londenberg
*/

#include "./dropout_allchannels-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(DropoutAllChannelsParam param) {
  return new DropoutAllChannelsOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet


