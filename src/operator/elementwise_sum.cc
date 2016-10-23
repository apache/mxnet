/*!
 * Copyright (c) 2015 by Contributors
 * \file elementwise_sum.cc
 * \brief elementwise sum operator
*/
#include "./elementwise_sum-inl.h"
#if MXNET_USE_MKL2017 == 1
#include <mxnet/mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_elementwise-inl.h"
#endif  // MXNET_USE_MKL2017

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ElementWiseSumParam param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_MKL2017 == 1
  switch (dtype) {
  case mshadow::kFloat32:
    op = new MKLElementWiseOp<cpu, float>(param, EltwiseParameter_EltwiseOp_SUM);
    break;
  case mshadow::kFloat64:
    op = new MKLElementWiseOp<cpu, double>(param, EltwiseParameter_EltwiseOp_SUM);
    break;
  default:
      break;
  }
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ElementWiseSumOp<cpu, DType>(param);
  });
#endif
  return op;
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* ElementWiseSumProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                               std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ElementWiseSumParam);

MXNET_REGISTER_OP_PROPERTY(ElementWiseSum, ElementWiseSumProp)
.describe("Perform an elementwise sum over all the inputs.")
.add_arguments(ElementWiseSumParam::__FIELDS__())
.set_key_var_num_args("num_args");

}  // namespace op
}  // namespace mxnet
