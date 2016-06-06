/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling.cc
 * \brief
 * \author Bing Xu
*/
#include "./pooling-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(PoolingParam param) {
  switch (param.pool_type) {
    case pool_enum::kMaxPooling:
      return new PoolingOp<cpu, mshadow::red::maximum>(param);
    case pool_enum::kAvgPooling:
      return new PoolingOp<cpu, mshadow::red::sum>(param);
    case pool_enum::kSumPooling:
      return new PoolingOp<cpu, mshadow::red::sum>(param);
    default:
      LOG(FATAL) << "unknown pooling type";
      return NULL;
  }
}

Operator* PoolingProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(PoolingParam);

MXNET_REGISTER_OP_PROPERTY(Pooling, PoolingProp)
.describe("Perform spatial pooling on inputs.")
.add_argument("data", "Symbol", "Input data to the pooling operator.")
.add_arguments(PoolingParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

