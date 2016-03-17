/*!
 * Copyright (c) 2015 by Contributors
 * \file cast.cc
 * \brief cast op
 * \author Junyuan Xie
*/
#include "./cast-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(CastParam param, std::vector<int> *in_type) {
  Operator *op = NULL;
  MSHADOW_TYPE_SWITCH((*in_type)[0], SrcDType, {
    MSHADOW_TYPE_SWITCH(param.dtype, DstDType, {
        op = new CastOp<cpu, SrcDType, DstDType>();
    })
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *CastProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type);
}

DMLC_REGISTER_PARAMETER(CastParam);

MXNET_REGISTER_OP_PROPERTY(Cast, CastProp)
.describe("Cast array to a different data type.")
.add_argument("data", "Symbol", "Input data to cast function.")
.add_arguments(CastParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

