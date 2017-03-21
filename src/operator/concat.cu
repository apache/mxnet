/*!
 * Copyright (c) 2015 by Contributors
 * \file concat.cu
 * \brief
 * \author Bing Xu
*/

#include "./concat-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ConcatParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    op = new ConcatOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet

