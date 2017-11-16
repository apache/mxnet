/*!
 * Copyright (c) 2015 by Contributors
 * \file warpctc.cc
 * \brief warpctc op
 * \author Liang Xiang
*/
#include "./warpctc-inl.h"
#include <stdio.h>
#include "../../src/operator/mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(WarpCTCParam param) {
  return new WarpCTCOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
