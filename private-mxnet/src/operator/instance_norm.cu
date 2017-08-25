/*!
 * Copyright (c) 2015 by Contributors
 * \file instance_norm.cu
 * \brief
 * \author Sebastian Bodenstein
*/

#include "./instance_norm-inl.h"

namespace mxnet {
namespace op {
template <>
Operator* CreateOp<gpu>(InstanceNormParam param, int dtype) {
  return new InstanceNormOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
