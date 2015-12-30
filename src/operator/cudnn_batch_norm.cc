/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_batch_norm.cc
 * \brief
 * \author Junyuan Xie
*/

#include "./cudnn_batch_norm-inl.h"
namespace mxnet {
namespace op {
#if CUDNN_MAJOR >= 4
template<>
Operator *CreateOp<cpu>(CuDNNBatchNormParam param) {
  LOG(FATAL) << "CuDNNBatchNormOp is only available for gpu.";
  return NULL;
}

Operator *CuDNNBatchNormProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(CuDNNBatchNormParam);

MXNET_REGISTER_OP_PROPERTY(CuDNNBatchNorm, CuDNNBatchNormProp)
.describe("Apply batch normalization to input.")
.add_argument("data", "Symbol", "Input data to batch normalization")
.add_arguments(CuDNNBatchNormParam::__FIELDS__());
#endif  // CUDNN_MAJOR >= 4
}  // namespace op
}  // namespace mxnet
