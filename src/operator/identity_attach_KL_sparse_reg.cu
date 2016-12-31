/*!
 * Copyright (c) 2015 by Contributors
 * \file identity_attach_KL_sparse_reg.cu
 * \brief
*/
#include "./identity_attach_KL_sparse_reg-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(IdentityAttachKLSparseRegParam param) {
  return new IdentityAttachKLSparseRegOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
