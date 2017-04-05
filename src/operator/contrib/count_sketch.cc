/*!
 * Copyright (c) 2015 by Contributors
 * \file count_sketch.cc
 * \brief count_sketch op
 * \author Chen Zhu
*/
#include "./count_sketch-inl.h"
namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(CountSketchParam param, int dtype) {
    LOG(FATAL) << "CountSketch is only available for GPU.";
    return NULL;
}
Operator *CountSketchProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(CountSketchParam);
MXNET_REGISTER_OP_PROPERTY(_contrib_count_sketch, CountSketchProp)
.describe("Apply CountSketch to input.")
.add_argument("data", "NDArray-or-Symbol", "Input data to the CountSketchOp.")
.add_argument("s", "NDArray-or-Symbol", "The sign vector")
.add_argument("h", "NDArray-or-Symbol", "The index vector")
.add_arguments(CountSketchParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
