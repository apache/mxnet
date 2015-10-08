/*!
 * Copyright (c) 2015 by Contributors
 * \file slice_channel.cc
 * \brief
 * \author Bing Xu
*/

#include "./slice_channel-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(SliceChannelParam param) {
  return new SliceChannelOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

