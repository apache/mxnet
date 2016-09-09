/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_loss.cc
 * \brief caffe loss 
 * \author Haoran Wang 
*/
#include "./caffe_loss-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(CaffeLossParam param, int dtype) {
  Operator *op = NULL;
  switch (dtype) {
  case mshadow::kFloat32:
    op = new CaffeLoss<cpu, float>(param);
    break;
  case mshadow::kFloat64:
    op = new CaffeLoss<cpu, double>(param);
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
Operator *CaffeLossProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
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

DMLC_REGISTER_PARAMETER(CaffeLossParam);

MXNET_REGISTER_OP_PROPERTY(CaffeLoss, CaffeLossProp)
.describe("Caffe loss layer")
.add_arguments(CaffeLossParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
