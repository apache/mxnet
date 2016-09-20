/*!
 * Copyright (c) 2015 by Contributors
 * \file sequence_last.cu
 * \brief
 * \author Sebastian Bodenstein
*/

#include "./sequence_last-inl.h"

namespace mxnet {
namespace op {
template <> Operator *CreateOp<gpu>(SequenceLastParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType,
                           { op = new SequenceLastOp<gpu, DType>(param); })
  return op;
}

}  // namespace op
}  // namespace mxnet
