/*!
 * Copyright (c) 2015 by Contributors
 * \file anchor_regressioncost.cu
 * \brief
 * \author Ming Zhang
*/

#include "./anchor_multibbs_regressioncost-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(AnchorMultiBBsRegCostParam param) {
  return new AnchorMultiBBsRegCostOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

