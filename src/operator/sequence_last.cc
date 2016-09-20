/*!
 * Copyright (c) 2015 by Contributors
 * \file sequence_last.cc
 * \brief
 * \author Sebastian Bodenstein
*/
#include "./sequence_last-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SequenceLastParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SequenceLastOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SequenceLastProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SequenceLastParam);

MXNET_REGISTER_OP_PROPERTY(SequenceLast, SequenceLastProp)
.describe("Get the last element of a sequence.")
.add_argument("data", "Symbol", "Input data to softmax.")
.add_argument("sequence_length", "Symbol", "vector of sequence lengths.")
.add_arguments(SequenceLastParam::__FIELDS__());


}  // namespace op
}  // namespace mxnet
