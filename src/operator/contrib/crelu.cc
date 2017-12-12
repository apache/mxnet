/*!
 * Copyright (c) 2017 by Contributors
 * \file crelu.cc
 * \brief crelu op
 * \author Yijie Zhuang
*/

#include "./crelu-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(CReluParam param, int dtype) {
  Operator *op = NULL;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CReluOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *CReluProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(CReluParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_CRelu, CReluProp)
.describe(R"code(Applies concatenate relu function to the input.

- `crelu`: Concatenate Rectified Linear Unit, :math:`y = (max(x, 0), max(-x,0))`

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array to activation function.")
.add_arguments(CReluParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_CRelu).add_alias("_contrib_crelu");

}  // namespace op
}  // namespace mxnet
