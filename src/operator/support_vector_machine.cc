/*!
 * Copyright (c) 2015 by Contributors
 * \file support_vector_machine.cc
 * \brief
 * \author Jonas Amaro
*/
#include "./support_vector_machine-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SupportVectorMachineParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SupportVectorMachineOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SupportVectorMachineProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SupportVectorMachineParam);

MXNET_REGISTER_OP_PROPERTY(SupportVectorMachine, SupportVectorMachineProp)
.describe("Support Vector Machine based transformation on input, backprop L2-SVM")
.add_argument("data", "Symbol", "Input data to svm.")
.add_argument("label", "Symbol", "Label data.")
.add_arguments(SupportVectorMachineParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

