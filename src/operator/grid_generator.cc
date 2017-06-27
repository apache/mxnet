/*!
 * Copyright (c) 2017 by Contributors
 * \file grid_generator.cc
 * \brief
 * \author Xu Dong
*/

#include "./grid_generator-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(GridGeneratorParam param, int dtype) {
  Operator *op = NULL;
  if (dtype == mshadow::kFloat32) {
    op = new GridGeneratorOp<cpu, float>(param);
  } else {
    LOG(FATAL) << "Other DTypes are not supported!";
  }
  return op;
}

Operator *GridGeneratorProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(GridGeneratorParam);

MXNET_REGISTER_OP_PROPERTY(GridGenerator, GridGeneratorProp)
.add_argument("data", "NDArray-or-Symbol", "Input data to the function.")
.add_arguments(GridGeneratorParam::__FIELDS__())
.describe("Generates 2D sampling grid for bilinear sampling.");

}  // namespace op
}  // namespace mxnet
