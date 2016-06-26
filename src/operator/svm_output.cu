/*!
 * Copyright (c) 2015 by Contributors
 * \file svm_output.cu
 * \brief
 * \author Jonas Amaro
*/

#include "./svm_output-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(SVMOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SVMOutputOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

