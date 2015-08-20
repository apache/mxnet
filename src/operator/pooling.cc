/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling.cc
 * \brief
 * \author Bing Xu
*/

#include <mxnet/registry.h>
#include "./pooling-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(PoolingParam param) {
  switch (param.type) {
    case kMaxPooling: return new PoolingOp<cpu, mshadow::red::maximum>(param);
    case kAvgPooling: return new PoolingOp<cpu, mshadow::red::sum>(param);
    case kSumPooling: return new PoolingOp<cpu, mshadow::red::sum>(param);
    default:
      LOG(FATAL) << "unknown activation type";
      return NULL;
  }
}

Operator* PoolingProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(PoolingParam);

REGISTER_OP_PROPERTY(Pooling, PoolingProp);
}  // namespace op
}  // namespace mxnet

