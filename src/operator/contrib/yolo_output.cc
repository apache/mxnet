/*!
 * Copyright (c) 2017 by Contributors
 * \file yolo_output.cc
 * \brief YoloOutput op
 * \author Joshua Zhang
*/
#include "./yolo_output-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(YoloOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new YoloOutputOp<cpu, DType>(param);
  });
  return op;
}

Operator *YoloOutputProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                                    std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(YoloOutputParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_YoloOutput, YoloOutputProp)
.describe("Yolo output layer.")
.add_argument("data", "NDArray-or-Symbol", "Input data to the YoloOutputOp.")
.add_argument("label", "NDArray-or-Symbol", "Object detection labels.")
.add_arguments(YoloOutputParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
