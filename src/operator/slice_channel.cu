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
Operator* CreateOp<gpu>(SliceChannelParam param, int dtype) {
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    return new SliceChannelOp<gpu>(param);
  })
}

}  // namespace op
}  // namespace mxnet

