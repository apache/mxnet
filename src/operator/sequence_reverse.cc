/*!
 * Copyright (c) 2015 by Contributors
 * \file sequence_reverse.cc
 * \brief
 * \author Sebastian Bodenstein
*/
#include "./sequence_reverse-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SequenceReverseParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SequenceReverseOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SequenceReverseProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SequenceReverseParam);

MXNET_REGISTER_OP_PROPERTY(SequenceReverse, SequenceReverseProp)
.describe("Get the last element of a sequence.")
.add_argument("data", "Symbol", "Input data of the form (seq len, other dims).")
.add_argument("sequence_length", "Symbol", "vector of sequence lengths.")
.add_arguments(SequenceReverseParam::__FIELDS__());


}  // namespace op
}  // namespace mxnet
