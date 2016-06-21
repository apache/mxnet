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
Operator *CreateOp_CuDNNv4<cpu>(BatchNormParam param, int dtype) {
  LOG(FATAL) << "CuDNNBatchNormOp is only available for gpu.";
  return NULL;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *CuDNNBatchNormProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
#if CUDNN_MAJOR >= 5
  LOG(FATAL) << "CuDNNBatchNorm is merged into BatchNorm for cudnn version above v5."
                "Use the later instead.";
  return nullptr;
#else
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp_CuDNNv4, param_, (*in_type)[0]);
  #endif
}

MXNET_REGISTER_OP_PROPERTY(CuDNNBatchNorm, CuDNNBatchNormProp)
.describe("Apply batch normalization to input.")
.add_argument("data", "Symbol", "Input data to batch normalization")
.add_arguments(BatchNormParam::__FIELDS__());
#endif  // CUDNN_MAJOR >= 4
}  // namespace op
}  // namespace mxnet
