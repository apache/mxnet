/*!
 * Copyright (c) 2015 by Contributors
 * \file make_loss.cu
 * \brief special layer for propagating loss
*/
#include "./make_loss-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(MakeLossParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MakeLossOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet

