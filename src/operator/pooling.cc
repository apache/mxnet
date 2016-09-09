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
Operator *CreateOp<cpu>(PoolingParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    switch (param.pool_type) {
      case pool_enum::kMaxPooling:
        op = new PoolingOp<cpu, mshadow::red::maximum, DType>(param);
        break;
      case pool_enum::kAvgPooling:
        op = new PoolingOp<cpu, mshadow::red::sum, DType>(param);
        break;
      case pool_enum::kSumPooling:
        op = new PoolingOp<cpu, mshadow::red::sum, DType>(param);
        break;
      default:
        LOG(FATAL) << "unknown pooling type";
        return NULL;
    }
  });
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator* PoolingProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(PoolingParam);

MXNET_REGISTER_OP_PROPERTY(Pooling, PoolingProp)
.describe("Perform spatial pooling on inputs.")
.add_argument("data", "Symbol", "Input data to the pooling operator.")
.add_arguments(PoolingParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

