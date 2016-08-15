/*!
 * Copyright (c) 2015 by Contributors
 * \file sequence_reverse.cu
 * \brief
 * \author Sebastian Bodenstein
*/

#include "./sequence_reverse-inl.h"

namespace mxnet {
namespace op {
template <> Operator *CreateOp<gpu>(SequenceReverseParam param, int dtype) {
  Operator *op = NULL;
  return op;
}

}  // namespace op
}  // namespace mxnet
