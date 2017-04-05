/*!
 * Copyright (c) 2015 by Contributors
 * \file sequence_mask.cc
 * \brief
 * \author Sebastian Bodenstein
*/
#include "./sequence_mask-inl.h"

namespace mshadow {

template <typename DType>
inline void SequenceMask(const Tensor<cpu, 3, DType> &dst,
                         const Tensor<cpu, 1, DType> label, DType value) {
  for (index_t b = 0; b < dst.size(1); ++b)
    for (index_t s = label[b]; s < dst.size(0); ++s)
      for (index_t r = 0; r < dst.size(2); ++r)
        dst[s][b][r] = value;
}

}  // namespace mshadow

namespace mxnet {
namespace op {
template <>
Operator *CreateOp<cpu>(SequenceMaskParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType,
                           { op = new SequenceMaskOp<cpu, DType>(param); })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SequenceMaskProp::CreateOperatorEx(Context ctx,
                                             std::vector<TShape> *in_shape,
                                             std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SequenceMaskParam);

MXNET_REGISTER_OP_PROPERTY(SequenceMask, SequenceMaskProp)
    .describe(
"Sets all elements outside the sequence to a constant value. Takes an n-dimensional tensor of the "
"form [max sequence length, batchsize, other dims] and returns a tensor of the same "
"shape. This operator takes an optional input tensor sequence_length of positive ints of "
"dimension [batchsize] when the sequence_length option is set to true. This allows the "
"operator to handle variable-length sequences. If sequence_length is false, then each "
"example in the batch is assumed to have the max sequence length, and this operator becomes "
"the identity operator."
)
    .add_argument("data", "NDArray-or-Symbol",
                  "n-dimensional input tensor of the form [max sequence "
                  "length, batchsize, other dims]")
    .add_argument("sequence_length", "NDArray-or-Symbol",
                  "vector of sequence lengths of size batchsize")
    .add_arguments(SequenceMaskParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
