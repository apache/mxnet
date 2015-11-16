/*!
 * Copyright (c) 2015 by Contributors
 * \file anchorclassifycost.cu
 * \brief
 * \author Ming Zhang
*/

#include "./anchorclassifycost-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(AnchorClsCostParam param) {
  return new AnchorClsCostOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

