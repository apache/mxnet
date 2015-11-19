/*!
 * Copyright (c) 2015 by Contributors
 * \file sparse_reg.cu
 * \brief
*/
#include "./sparse_reg-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(SparseRegParam param) {
  return new SparseRegOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
