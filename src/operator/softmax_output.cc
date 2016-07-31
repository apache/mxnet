/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_output.cc
 * \brief
 * \author Bing Xu
*/
#include "./softmax_output-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SoftmaxOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SoftmaxOutputOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SoftmaxOutputProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SoftmaxOutputParam);

MXNET_REGISTER_OP_PROPERTY(SoftmaxOutput, SoftmaxOutputProp)
.describe("Perform a softmax transformation on input, backprop with logloss.")
.add_argument("data", "Symbol", "Input data to softmax.")
.add_argument("label", "Symbol", "Label data, can also be "\
              "probability value with same shape as data")
.add_arguments(SoftmaxOutputParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(Softmax, DeprecatedSoftmaxProp)
.describe("DEPRECATED: Perform a softmax transformation on input. Please use SoftmaxOutput")
.add_argument("data", "Symbol", "Input data to softmax.")
.add_arguments(SoftmaxOutputParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
