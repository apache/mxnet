/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_operator.cc
 * \brief caffe operator
 * \author Haoran Wang 
*/
#include "./caffe_operator-inl.h"
#include "./caffe_operator_util.h"
namespace mxnet {
namespace op {

template<>
Operator* CreateOp<cpu>(CaffeOperatorParam param, int dtype) {
  printf("Caffe Op: param\n");
  Operator *op = NULL;
  switch (dtype) {
  case mshadow::kFloat32:
    op = new CaffeOperator<cpu, float>(param);
    break;
  case mshadow::kFloat64:
    op = new CaffeOperator<cpu, double>(param);
    break;
  case mshadow::kFloat16:
    LOG(FATAL) << "float16 layer is not supported by caffe";
    break;
  default:
    LOG(FATAL) << "Unsupported type " << dtype;
  }
  return op;
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator *CaffeOperatorProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  printf("CaffeOperatorProp::CreateOperatorEx\n");
  std::vector<int> out_type, aux_type;
  std::vector<TShape> out_shape, aux_shape;
  out_type.resize(this->ListOutputs().size());
  out_shape.resize(this->ListOutputs().size());
  aux_type.resize(this->ListAuxiliaryStates().size());
  aux_shape.resize(this->ListAuxiliaryStates().size());
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(CaffeOperatorParam);

MXNET_REGISTER_OP_PROPERTY(CaffeOperator, CaffeOperatorProp)
.describe("Apply caffe operator")
.add_argument("data", "Symbol[]", "List of tensors")
.add_arguments(CaffeOperatorParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
