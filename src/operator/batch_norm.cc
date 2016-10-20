/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm.cc
 * \brief
 * \author Bing Xu
*/

#include "./batch_norm-inl.h"
#if MXNET_USE_MKLDNN == 1
#include "./mkldnn/mkldnn_batch_norm-inl.h"
#endif

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(BatchNormParam param) {
#if MXNET_USE_MKLDNN == 1
  return new MKLBatchNormOp<float>(param);
#else
  return new BatchNormOp<cpu>(param);
#endif
}

Operator *BatchNormProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(BatchNormParam);

MXNET_REGISTER_OP_PROPERTY(BatchNorm, BatchNormProp)
.describe("Apply batch normalization to input.")
.add_argument("data", "Symbol", "Input data to batch normalization")
.add_arguments(BatchNormParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

