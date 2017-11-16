/*!
 * Copyright (c) 2015 by Contributors
 * \file sequence_mask.cu
 * \brief
 * \author Sebastian Bodenstein
*/

#include "./sequence_mask-inl.h"

namespace mxnet {
namespace op {

template <> Operator *CreateOp<gpu>(SequenceMaskParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType,
                           { op = new SequenceMaskOp<gpu, DType>(param); })
  return op;
}

}  // namespace op
}  // namespace mxnet
