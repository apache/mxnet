/*!
 * Copyright (c) 2015 by Contributors
 * \file concat.cc
 * \brief
 * \author Bing Xu
*/

#include "./concat-inl.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_concat-inl.h"
#endif  // MXNET_USE_MKL2017

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ConcatParam param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_MKL2017 == 1
  if (1 == param.dim) {
    switch (dtype) {
      case mshadow::kFloat32:
      return new MKLConcatOp<cpu, float>(param);
    case mshadow::kFloat64:
      return new MKLConcatOp<cpu, double>(param);
    default:
      break;
    }
  }
  LOG(INFO) << MKLConcatOp<cpu, float>::getName() << " Skip MKL optimization";
#endif
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ConcatOp<cpu, DType>(param);
  });
  return op;
}

Operator* ConcatProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                       std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ConcatParam);

MXNET_REGISTER_OP_PROPERTY(Concat, ConcatProp)
.add_argument("data", "Symbol[]", "List of tensors to concatenate")
.add_arguments(ConcatParam::__FIELDS__())
.set_key_var_num_args("num_args")
.describe("Perform an feature concat on channel dim (defaut is 1) over all");

}  // namespace op
}  // namespace mxnet
