
/*!
 * Copyright (c) 2017 by Contributors
 * \file crelu.cu
 * \brief crelu op
 * \author Yijie Zhuang
*/

#include "./crelu-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(CReluParam param, int dtype) {
  Operator *op = NULL;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CReluOp<gpu, DType>(param);
  })

  return op;
}

}  // namespace op
}  // namespace mxnet
