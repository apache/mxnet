/*!
 * Copyright (c) 2015 by Contributors
 * \file lrn.cc
 * \brief
 * \author Bing Xu
*/

#include "./lrn-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_lrn-inl.h"
#endif

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(LRNParam param) {
  return new LocalResponseNormOp<cpu>(param);
}

Operator* LocalResponseNormProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(LRNParam);

MXNET_REGISTER_OP_PROPERTY(LRN, LocalResponseNormProp)
.add_argument("data", "Symbol", "Input data to the ConvolutionOp.")
.add_arguments(LRNParam::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet
